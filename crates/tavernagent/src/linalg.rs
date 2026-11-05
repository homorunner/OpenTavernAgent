//! Row-major f32 matrices with just the operations the network needs.
//!
//! Matrix products go through `matrixmultiply::sgemm`; everything else is a plain loop.
//! A `Mat` with one row doubles as a vector.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn filled(rows: usize, cols: usize, value: f32) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Mat {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrix data does not match its shape"
        );
        Mat { rows, cols, data }
    }

    pub fn zeros_like(&self) -> Mat {
        Mat::zeros(self.rows, self.cols)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// `self (m x k) * other (k x n)`.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        let mut out = Mat::zeros(self.rows, other.cols);
        gemm(false, self, false, other, &mut out);
        out
    }

    /// `self^T (k x m)^T * other (k x n)`: gradient of a weight from inputs and output grads.
    pub fn t_matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.rows, other.rows, "t_matmul shape mismatch");
        let mut out = Mat::zeros(self.cols, other.cols);
        gemm(true, self, false, other, &mut out);
        out
    }

    /// `self (m x n) * other^T (k x n)^T`: gradient of an input from output grads and weights.
    pub fn matmul_t(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.cols, "matmul_t shape mismatch");
        let mut out = Mat::zeros(self.rows, other.rows);
        gemm(false, self, true, other, &mut out);
        out
    }

    /// `x (1 x k) * self (k x n)` for a single row, skipping zero inputs: the observation
    /// is sparse and this path serves every search evaluation.
    pub fn vecmat(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.rows, "vecmat shape mismatch");
        let mut out = vec![0.0f32; self.cols];
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = self.row(i);
            for (o, w) in out.iter_mut().zip(row) {
                *o += xi * w;
            }
        }
        out
    }

    /// Add a row vector to every row.
    pub fn add_row(&mut self, bias: &[f32]) {
        assert_eq!(bias.len(), self.cols);
        for r in 0..self.rows {
            for (x, b) in self.row_mut(r).iter_mut().zip(bias) {
                *x += b;
            }
        }
    }

    /// Column sums as a vector (gradient of a bias).
    pub fn sum_rows(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.cols];
        for r in 0..self.rows {
            for (o, x) in out.iter_mut().zip(self.row(r)) {
                *o += x;
            }
        }
        out
    }

    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    pub fn scale(&mut self, k: f32) {
        for x in &mut self.data {
            *x *= k;
        }
    }

    pub fn sum_squares(&self) -> f64 {
        self.data.iter().map(|x| (*x as f64) * (*x as f64)).sum()
    }

    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }
}

/// Round-to-nearest-even conversion to bfloat16 (the top 16 bits of an f32).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Row-major bfloat16 copy of a weight matrix for search-time products: half the bytes
/// of `Mat`, so the weights of a 192-wide network stay in a core's L2 cache; products
/// accumulate in f32. Training never uses it.
#[derive(Clone, Debug)]
pub struct Bf16Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u16>,
}

impl Bf16Mat {
    pub fn from_mat(m: &Mat) -> Bf16Mat {
        Bf16Mat {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(|&v| f32_to_bf16(v)).collect(),
        }
    }

    /// `x (1 x k) * self (k x n)` for a single row, skipping zero inputs.
    pub fn vecmat(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.rows, "vecmat shape mismatch");
        let mut out = vec![0.0f32; self.cols];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
            {
                // SAFETY: the features were detected at runtime.
                unsafe { vecmat_bf16_avx512(&self.data, self.cols, x, &mut out) };
                return out;
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                // SAFETY: the features were detected at runtime.
                unsafe { vecmat_bf16_avx2(&self.data, self.cols, x, &mut out) };
                return out;
            }
        }
        vecmat_bf16_rows(&self.data, self.cols, x, &mut out);
        out
    }
}

/// The product kernel: for every nonzero input, add its bf16 weight row into `out`.
/// Written as plain loops so that the compiler vectorises it for the enabled features.
#[inline(always)]
fn vecmat_bf16_rows(data: &[u16], cols: usize, x: &[f32], out: &mut [f32]) {
    let out = &mut out[..cols];
    for (i, &xi) in x.iter().enumerate() {
        if xi == 0.0 {
            continue;
        }
        let row = &data[i * cols..(i + 1) * cols];
        for (o, &w) in out.iter_mut().zip(row) {
            *o += xi * bf16_to_f32(w);
        }
    }
}

/// AVX-512 kernel: up to `ROWS` nonzero inputs per pass, the 16-wide output blocks kept in
/// registers across those rows, bf16 widened by a 16-bit shift. Needs avx512f/bw/vl.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn vecmat_bf16_avx512(data: &[u16], cols: usize, x: &[f32], out: &mut [f32]) {
    // SAFETY: the caller checked the CPU features; every pointer stays inside `data`,
    // `x` and `out` by the shape assertion in `Bf16Mat::vecmat`.
    unsafe {
        use std::arch::x86_64::*;
        const ROWS: usize = 8;
        debug_assert!(out.len() >= cols && data.len() >= x.len() * cols);
        let full = cols - cols % 16;
        let tail_mask: __mmask16 = ((1u32 << (cols - full)) - 1) as __mmask16;
        let mut idx = [0usize; ROWS];
        let mut xb = [_mm512_setzero_ps(); ROWS];
        let mut i = 0;
        while i < x.len() {
            let mut k = 0;
            while i < x.len() && k < ROWS {
                let xi = x[i];
                if xi != 0.0 {
                    idx[k] = i;
                    xb[k] = _mm512_set1_ps(xi);
                    k += 1;
                }
                i += 1;
            }
            if k == 0 {
                break;
            }
            let mut c = 0;
            while c < full {
                let mut acc = _mm512_loadu_ps(out.as_ptr().add(c));
                for r in 0..k {
                    let w =
                        _mm256_loadu_si256(data.as_ptr().add(idx[r] * cols + c) as *const __m256i);
                    let w = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w));
                    acc = _mm512_fmadd_ps(xb[r], _mm512_castsi512_ps(w), acc);
                }
                _mm512_storeu_ps(out.as_mut_ptr().add(c), acc);
                c += 16;
            }
            if full < cols {
                let mut acc = _mm512_maskz_loadu_ps(tail_mask, out.as_ptr().add(full));
                for r in 0..k {
                    let w = _mm256_maskz_loadu_epi16(
                        tail_mask,
                        data.as_ptr().add(idx[r] * cols + full) as *const i16,
                    );
                    let w = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w));
                    acc = _mm512_fmadd_ps(xb[r], _mm512_castsi512_ps(w), acc);
                }
                _mm512_mask_storeu_ps(out.as_mut_ptr().add(full), tail_mask, acc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vecmat_bf16_avx2(data: &[u16], cols: usize, x: &[f32], out: &mut [f32]) {
    vecmat_bf16_rows(data, cols, x, out)
}

fn gemm(ta: bool, a: &Mat, tb: bool, b: &Mat, c: &mut Mat) {
    let (m, k) = if ta {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let n = if tb { b.rows } else { b.cols };
    debug_assert_eq!(c.rows, m);
    debug_assert_eq!(c.cols, n);
    // Row-major strides; a transposed operand swaps its strides.
    let (rsa, csa) = if ta {
        (1isize, a.cols as isize)
    } else {
        (a.cols as isize, 1isize)
    };
    let (rsb, csb) = if tb {
        (1isize, b.cols as isize)
    } else {
        (b.cols as isize, 1isize)
    };
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    // SAFETY: the dimensions and strides above describe exactly the buffers of a, b and c.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.data.as_ptr(),
            rsa,
            csa,
            b.data.as_ptr(),
            rsb,
            csb,
            0.0,
            c.data.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64Mcg;

    fn naive(a: &Mat, b: &Mat) -> Mat {
        let mut c = Mat::zeros(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut s = 0.0;
                for k in 0..a.cols {
                    s += a.data[i * a.cols + k] * b.data[k * b.cols + j];
                }
                c.data[i * b.cols + j] = s;
            }
        }
        c
    }

    fn close(a: &Mat, b: &Mat) {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    fn transpose(a: &Mat) -> Mat {
        let mut t = Mat::zeros(a.cols, a.rows);
        for i in 0..a.rows {
            for j in 0..a.cols {
                t.data[j * a.rows + i] = a.data[i * a.cols + j];
            }
        }
        t
    }

    #[test]
    fn bf16_round_trips_and_products_stay_close() {
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        assert_eq!(bf16_to_f32(f32_to_bf16(-0.5)), -0.5);
        // Round to nearest even: 1 + 2^-8 is halfway between two bf16 values.
        let x = 1.0 + 2f32.powi(-8);
        assert_eq!(bf16_to_f32(f32_to_bf16(x)), 1.0);
        let y = 1.0 + 3.0 * 2f32.powi(-8);
        assert_eq!(bf16_to_f32(f32_to_bf16(y)), 1.0 + 2f32.powi(-6));
        for v in [0.1f32, -3.7, 1e-3, 250.0] {
            assert!(((bf16_to_f32(f32_to_bf16(v)) - v) / v).abs() <= 2f32.powi(-8));
        }
        let mut rng = Pcg64Mcg::seed_from_u64(9);
        let w = Mat::from_vec(
            300,
            192,
            (0..300 * 192).map(|_| rng.random::<f32>() - 0.5).collect(),
        );
        let x: Vec<f32> = (0..300)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    rng.random::<f32>() * 2.0 - 1.0
                }
            })
            .collect();
        let exact = w.vecmat(&x);
        let fast = Bf16Mat::from_mat(&w).vecmat(&x);
        let scale = exact.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (a, b) in exact.iter().zip(&fast) {
            assert!((a - b).abs() < 0.01 * scale, "{a} vs {b}");
        }
    }

    #[test]
    fn vecmat_matches_matmul() {
        let w = Mat::from_vec(5, 3, (0..15).map(|i| i as f32 * 0.3 - 1.0).collect());
        let x = Mat::from_vec(1, 5, vec![1.0, 0.0, -2.0, 0.0, 0.5]);
        assert_eq!(w.vecmat(x.row(0)), x.matmul(&w).data);
    }

    #[test]
    fn products_match_naive_loops() {
        let a = Mat::from_vec(3, 4, (0..12).map(|i| i as f32 * 0.5 - 2.0).collect());
        let b = Mat::from_vec(4, 5, (0..20).map(|i| (i * 7 % 11) as f32 - 5.0).collect());
        close(&a.matmul(&b), &naive(&a, &b));
        let g = Mat::from_vec(3, 5, (0..15).map(|i| i as f32 * 0.1).collect());
        close(&a.t_matmul(&g), &naive(&transpose(&a), &g));
        close(&g.matmul_t(&b), &naive(&g, &transpose(&b)));
    }
}
