//! Row-major f32 matrices with just the operations the network needs.
//!
//! Matrix products go through `matrixmultiply::sgemm`; everything else is a plain loop.
//! A `Mat` with one row doubles as a vector.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
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

    /// Reuse storage; callers must overwrite existing entries before reading them.
    pub(crate) fn resize(&mut self, rows: usize, cols: usize) {
        self.data.resize(rows * cols, 0.0);
        self.rows = rows;
        self.cols = cols;
    }

    pub(crate) fn matmul_into(&self, other: &Mat, out: &mut Mat) {
        assert_eq!(self.cols, other.rows, "matmul shape mismatch");
        out.resize(self.rows, other.cols);
        if self.cols == 0 {
            out.data.fill(0.0);
        }
        gemm(false, self, false, other, out);
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

    /// Multiply by a contiguous range of weight rows, optionally transposed.
    /// Borrows the weights directly instead of allocating a partition copy.
    pub(crate) fn matmul_weight_rows(
        &self,
        weights: &Mat,
        start: usize,
        rows: usize,
        transpose: bool,
    ) -> Mat {
        let mut out = Mat::default();
        self.matmul_weight_rows_into(weights, start, rows, transpose, &mut out);
        out
    }

    pub(crate) fn matmul_weight_rows_into(
        &self,
        weights: &Mat,
        start: usize,
        rows: usize,
        transpose: bool,
        out: &mut Mat,
    ) {
        assert!(start <= weights.rows && rows <= weights.rows - start);
        let (k, n, rs, cs) = if transpose {
            (weights.cols, rows, 1, weights.cols as isize)
        } else {
            (rows, weights.cols, weights.cols as isize, 1)
        };
        assert_eq!(self.cols, k);
        out.resize(self.rows, n);
        if k == 0 {
            out.data.fill(0.0);
        }
        if self.rows != 0 && k != 0 && n != 0 {
            // SAFETY: checked shapes and row bounds cover both contiguous inputs
            // and the output; transposition only exchanges the weight strides.
            unsafe {
                matrixmultiply::sgemm(
                    self.rows,
                    k,
                    n,
                    1.0,
                    self.data.as_ptr(),
                    self.cols as isize,
                    1,
                    weights.data[start * weights.cols..].as_ptr(),
                    rs,
                    cs,
                    0.0,
                    out.data.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        }
    }

    /// `x (1 x k) * self (k x n)` for a single row, skipping zero inputs: the observation
    /// is sparse and this path serves every rollout decision.
    pub fn vecmat(&self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.vecmat_into(x, &mut out);
        out
    }

    pub(crate) fn vecmat_into(&self, x: &[f32], out: &mut Vec<f32>) {
        assert_eq!(x.len(), self.rows, "vecmat shape mismatch");
        out.resize(self.cols, 0.0);
        out.fill(0.0);
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let row = self.row(i);
            for (o, w) in out.iter_mut().zip(row) {
                *o += xi * w;
            }
        }
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
