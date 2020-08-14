//use std::error::Error;
use crate::exp_approx;
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_broadcast_ss, _mm256_mul_ps, _mm256_setzero_ps,
};
use std::mem::transmute;

const SINGLES_PER_AVX: usize = 8;

// matrix of f32, but we split the supplied rows into
// columns of AVX instrinsics (8 x 32-bit floats), and then
// do a column-wise multiplication
pub struct MatrixAvxF32 {
    pub num_columns: usize,
    pub num_col_instrinsics: usize,
    pub num_rows: usize,
    column_intrinsics: Vec<Vec<__m256>>,
    intercept_intrinsics: Vec<__m256>,
}

impl MatrixAvxF32 {
    pub fn create_from_rows(rows: &Vec<Vec<f32>>, intercepts: &[f32]) -> Option<Self> {
        let num_columns = rows.first()?.len();
        if num_columns != intercepts.len() {
            return None;
        }

        let num_col_instrinsics = (num_columns / SINGLES_PER_AVX)
            + match num_columns % SINGLES_PER_AVX {
                0 => 0,
                _ => 1,
            };

        let mut mat = Self {
            num_columns,
            num_col_instrinsics,
            num_rows: rows.len(),
            column_intrinsics: vec![],
            intercept_intrinsics: vec![unsafe { _mm256_setzero_ps() }; num_col_instrinsics],
        };

        // copy intercepts
        for (intercept_chunk, dest) in intercepts
            .chunks(SINGLES_PER_AVX)
            .zip(mat.intercept_intrinsics.iter_mut())
        {
            let dest_cast: &mut [f32; SINGLES_PER_AVX] = unsafe { transmute(dest) };
            dest_cast[..intercept_chunk.len()].copy_from_slice(intercept_chunk);
        }

        // copy coefficients
        for chunk_num in 0..num_col_instrinsics {
            let mut col: Vec<__m256> = Vec::new();
            for r in rows {
                let chunk = r.chunks(SINGLES_PER_AVX).nth(chunk_num)?;
                let mut packed = unsafe { _mm256_setzero_ps() };
                let intrin =
                    unsafe { transmute::<&mut __m256, &mut [f32; SINGLES_PER_AVX]>(&mut packed) };
                intrin.iter_mut().zip(chunk).for_each(|(v, u)| *v = *u);
                col.push(packed);
            }
            mat.column_intrinsics.push(col);
        }

        Some(mat)
    }

    pub fn product(&self, values: &[f32], destination: &mut [f32]) -> Option<()> {
        if destination.len() != self.num_columns || values.len() != self.num_rows {
            return None;
        }

        destination
            .chunks_mut(SINGLES_PER_AVX)
            .zip(self.column_intrinsics.iter())
            .zip(self.intercept_intrinsics.iter())
            .for_each(|((dst, col), intercepts)| {
                // run multiplication and add to `accumulate`
                let mut accumulate = *intercepts;
                for (val, row_intrin) in values.iter().zip(col) {
                    unsafe {
                        // broadcast value (since we already have a reference)
                        let val_broad = _mm256_broadcast_ss(val);
                        let mult = _mm256_mul_ps(val_broad, *row_intrin);
                        accumulate = _mm256_add_ps(accumulate, mult);
                    }
                }
                // copy to destination (by interpreting the intrinsic as a slice) -- and we might
                // have a shorter final slice
                let src = unsafe { transmute::<&__m256, &[f32; 8]>(&accumulate) };
                dst.copy_from_slice(&src[0..(dst.len())]);
            });

        Some(())
    }

    pub fn product_mask_store(&self, values: &[f32], destination: &mut [f32]) -> Option<()> {
        use std::arch::x86_64::*;

        // TODO: Use AVX2 -- kind of hard to test at home
        let idx_lo = unsafe { _mm_set_epi32(3, 2, 1, 0) };
        let idx_hi = unsafe { _mm_set_epi32(7, 6, 5, 4) };

        if destination.len() != self.num_columns || values.len() != self.num_rows {
            return None;
        }

        destination
            .chunks_mut(SINGLES_PER_AVX)
            .zip(self.column_intrinsics.iter())
            .zip(self.intercept_intrinsics.iter())
            .for_each(|((dst, col), intercepts)| {
                // run multiplication and add to `accumulate`
                let mut accumulate = *intercepts;
                for (val, row_intrin) in values.iter().zip(col) {
                    unsafe {
                        // broadcast value (since we already have a reference)
                        let val_broad = _mm256_broadcast_ss(val);
                        let mult = _mm256_mul_ps(val_broad, *row_intrin);
                        accumulate = _mm256_add_ps(accumulate, mult);
                    }
                }

                // copy to destination -- and we might have a shorter final slice,
                // so use a mask to do the store
                unsafe {
                    // prep storage mask; TODO: AVX2
                    let length = _mm_set1_epi32(dst.len() as i32);
                    let lo_mask = _mm_cmplt_epi32(idx_lo, length);
                    let hi_mask = _mm_cmplt_epi32(idx_hi, length);
                    let mask = _mm256_set_m128i(hi_mask, lo_mask);
                    // and store using mask
                    let ptr_dest: *mut f32 = transmute(dst.as_mut_ptr());
                    _mm256_maskstore_ps(ptr_dest, mask, accumulate);
                }
            });

        Some(())
    }

    pub fn product_softmax_cumulative_approx(
        &self,
        values: &[f32],
        destination: &mut [f32],
    ) -> Option<()> {
        if destination.len() != self.num_columns || values.len() != self.num_rows {
            return None;
        }

        let mut cumulative_sum = 0f32;

        destination
            .chunks_mut(SINGLES_PER_AVX)
            .zip(self.column_intrinsics.iter())
            .zip(self.intercept_intrinsics.iter())
            .for_each(|((dst, col), intercepts)| {
                // run multiplication and add to `accumulate`, starting with the intercepts
                let mut accumulate = *intercepts;
                for (val, row_intrin) in values.iter().zip(col) {
                    unsafe {
                        // broadcast value (since we already have a reference)
                        let val_broad = _mm256_broadcast_ss(val);
                        let mult = _mm256_mul_ps(val_broad, *row_intrin);
                        accumulate = _mm256_add_ps(accumulate, mult);
                    }
                }

                // copy to destination (taking into account final shorter stub) and apply cumulative softmax
                // 1. approximate exponential
                accumulate = exp_approx::exp_approx_avxf32(accumulate);
                // 2. accumulate and copy
                let src = unsafe { transmute::<&__m256, &[f32; 8]>(&accumulate) };
                dst.iter_mut().zip(src).for_each(|(d, s)| {
                    cumulative_sum += s;
                    *d = cumulative_sum;
                });
            });

        Some(())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn structure_create_exact() {

        // TODO -- add intercepts and check

        // let rows = vec![
        //     vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
        //     vec![9f32, 10., 11., 12., 13., 14., 15., 16.],
        // ];

        // let cf = crate::MatrixAvxF32ColumnSets::create_from_rows(&rows).unwrap();

        // TODO

        // assert_eq!(cf.num_columns, 8);
        // assert_eq!(cf.num_rows, 2);
        // assert_eq!(cf.rows.len(), 2);
        // assert_eq!(cf.rows[0].len(), 1);
        // assert_eq!(cf.rows[1].len(), 1);
    }

    #[test]
    fn product() {
        let rows = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];
        let intercepts = [10f32, 20f32, 30f32];
        let matrix = super::MatrixAvxF32::create_from_rows(&rows, &intercepts).unwrap();
        let v = vec![1f32, 2.];
        let mut res = vec![0f32; 3];
        matrix.product(&v, &mut res);
        assert_eq!(res, [19f32, 32., 45.]);
    }

    #[test]
    fn product_large() {
        let coeffs: Vec<f32> = (1..=(35 * 5)).map(|x| x as f32).collect();
        let rows: Vec<Vec<f32>> = coeffs[..].chunks(35).map(|c| c.to_vec()).collect();
        let intercepts = [0f32; 35]; // leave these zero; another test covers this

        let matrix = super::MatrixAvxF32::create_from_rows(&rows, &intercepts).unwrap();
        let v: Vec<f32> = (1..=5).map(|x| x as f32).collect();

        // output to f32
        let mut res = vec![0f32; 35];
        matrix.product(&v, &mut res);
        assert_eq!(res[0], 1415_f32);
        assert_eq!(res[18], 1685_f32);
        assert_eq!(res[34], 1925_f32);
    }
}
