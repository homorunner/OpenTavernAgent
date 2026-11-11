// Embedded PTX kernels; compile without --use_fast_math and with --fmad=false.
// Matrix products use cuBLAS in pedantic FP32 mode. No reduced-precision inputs.
#include <math.h>

__device__ float activate(float x, int kind) {
    if (kind == 1) return x * (0.5f * (1.f + tanhf(0.5f * x)));
    if (kind == 2) return fmaxf(x, 0.f);
    float z = 0.7978846f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(z));
}
__device__ float derivative(float x, int kind) {
    if (kind == 1) { float s = 0.5f * (1.f + tanhf(0.5f*x)); return s+x*s*(1.f-s); }
    if (kind == 2) return x > 0.f ? 1.f : 0.f;
    float z = 0.7978846f * (x + 0.044715f*x*x*x), t = tanhf(z);
    return 0.5f*(1.f+t)+0.5f*x*(1.f-t*t)*(0.7978846f*(1.f+3.f*0.044715f*x*x));
}
extern "C" __global__ void bias(float* x, const float* b, int n, int cols) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=b[i%cols];
}
extern "C" __global__ void activation(const float* x, float* y, int n, int kind) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=activate(x[i],kind);
}
extern "C" __global__ void activation_grad(float* dx, const float* x, int n, int kind) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dx[i]*=derivative(x[i],kind);
}
extern "C" __global__ void axpy(float* dst, const float* src, int n, float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]+=scale*src[i];
}
extern "C" __global__ void scale(float* x, int n, float factor) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=factor;
}
extern "C" __global__ void value_tanh(float* x, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=tanhf(x[i]);
}
extern "C" __global__ void column_sum(const float* x, float* out, int rows, int cols) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<cols) { float s=0; for(int r=0;r<rows;r++) s+=x[r*cols+j]; out[j]=s; }
}
__device__ float reduce(float x, float* shared) {
    int t=threadIdx.x; shared[t]=x; __syncthreads();
    for(int s=128;s>0;s/=2) { if(t<s) shared[t]+=shared[t+s]; __syncthreads(); }
    float result=shared[0]; __syncthreads(); return result;
}
extern "C" __global__ void layer_norm(const float* x, const float* g, const float* b,
    float* y, float* xhat, float* inv, int rows, int cols) {
    __shared__ float scratch[256]; int r=blockIdx.x,t=threadIdx.x;
    if(r>=rows) return;
    float sum=0; for(int j=t;j<cols;j+=256) sum+=x[r*cols+j];
    float mean=reduce(sum,scratch)/(float)cols;
    sum=0; for(int j=t;j<cols;j+=256) { float d=x[r*cols+j]-mean; sum+=d*d; }
    float inverse=1.f/sqrtf(reduce(sum,scratch)/(float)cols+1e-5f);
    if(t==0) inv[r]=inverse;
    for(int j=t;j<cols;j+=256) { float v=(x[r*cols+j]-mean)*inverse;
        xhat[r*cols+j]=v; y[r*cols+j]=v*(g?g[j]:1.f)+(b?b[j]:0.f); }
}
extern "C" __global__ void layer_norm_dx(const float* dy,const float* xhat,const float* inv,
    const float* g,float* dx,int rows,int cols) {
    __shared__ float scratch[256]; int r=blockIdx.x,t=threadIdx.x;
    if(r>=rows) return;
    float a=0,b=0;
    for(int j=t;j<cols;j+=256) { float v=dy[r*cols+j]*(g?g[j]:1.f); a+=v; b+=v*xhat[r*cols+j]; }
    a=reduce(a,scratch)/(float)cols; b=reduce(b,scratch)/(float)cols;
    for(int j=t;j<cols;j+=256) dx[r*cols+j]=inv[r]*(dy[r*cols+j]*(g?g[j]:1.f)-a-xhat[r*cols+j]*b);
}
extern "C" __global__ void layer_norm_params(const float* dy,const float* xhat,
    float* dg,float* db,int rows,int cols) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<cols) { float a=0,b=0; for(int r=0;r<rows;r++){a+=dy[r*cols+j]*xhat[r*cols+j];b+=dy[r*cols+j];}
        dg[j]=a;db[j]=b; }
}
extern "C" __global__ void embed(float* x,const int* cards,const float* weights,
    int rows,int cols,int numeric,int slots,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*slots*dim) { int r=i/(slots*dim),s=(i/dim)%slots,d=i%dim,id=cards[r*slots+s];
        x[r*cols+numeric+s*dim+d]=id ? weights[id*dim+d] : 0.f; }
}
// Ordered per-card occurrence lists avoid nondeterministic floating-point atomics.
extern "C" __global__ void embed_grad(const float* dx,const int* offsets,const int* positions,
    float* grad,int vocabulary,int cols,int numeric,int slots,int dim) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<vocabulary*dim) { int id=i/dim,d=i%dim; float sum=0;
        if(id) for(int j=offsets[id];j<offsets[id+1];j++) { int p=positions[j];
            sum+=dx[(p/slots)*cols+numeric+(p%slots)*dim+d]; }
        grad[i]=sum; }
}
extern "C" __global__ void sum_squares(const float* x,double* parts,int n) {
    __shared__ double scratch[256]; int t=threadIdx.x,i=blockIdx.x*256+t;
    double v=i<n?(double)x[i]:0.; scratch[t]=v*v; __syncthreads();
    for(int s=128;s>0;s/=2){if(t<s)scratch[t]+=scratch[t+s];__syncthreads();}
    if(t==0) parts[blockIdx.x]=scratch[0];
}
extern "C" __global__ void adam(float* p,float* m,float* v,const float* grad,int n,
    double scale,double bias1,double bias2,double lr,double decay) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){double g=(double)grad[i]*scale;
        m[i]=(float)(0.9*(double)m[i]+(1.-0.9)*g);
        v[i]=(float)(0.999*(double)v[i]+(1.-0.999)*g*g);
        double update=((double)m[i]/bias1)/(sqrt((double)v[i]/bias2)+1e-8)+decay*(double)p[i];
        p[i]=(float)((double)p[i]-lr*update);}
}
