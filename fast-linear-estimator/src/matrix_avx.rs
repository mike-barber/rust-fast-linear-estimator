use std::arch::x86_64::*;
use std::mem::transmute;

pub const SINGLES_PER_INTRINSIC: usize = 8;

// matrix of f32, but we split the supplied rows into
// columns of AVX instrinsics (8 x 32-bit floats), and then
// do a column-wise multiplication
pub struct MatrixF32 {
    pub num_columns: usize,
    pub num_col_instrinsics: usize,
    pub num_rows: usize,
    column_intrinsics: Vec<Vec<__m256>>,
    intercept_intrinsics: Vec<__m256>,
}

pub fn zeros() -> __m256 {
    unsafe { std::mem::transmute([0f32; SINGLES_PER_INTRINSIC]) }
}

impl MatrixF32 {
    pub fn create_from_rows(rows: &Vec<Vec<f32>>, intercepts: &[f32]) -> Option<Self> {
        let num_columns = rows.first()?.len();
        if num_columns != intercepts.len() {
            return None;
        }

        let num_col_instrinsics = (num_columns / SINGLES_PER_INTRINSIC)
            + match num_columns % SINGLES_PER_INTRINSIC {
                0 => 0,
                _ => 1,
            };

        let mut mat = Self {
            num_columns,
            num_col_instrinsics,
            num_rows: rows.len(),
            column_intrinsics: vec![],
            intercept_intrinsics: vec![zeros(); num_col_instrinsics],
        };

        // copy intercepts
        for (intercept_chunk, dest) in intercepts
            .chunks(SINGLES_PER_INTRINSIC)
            .zip(mat.intercept_intrinsics.iter_mut())
        {
            let dest_cast: &mut [f32; SINGLES_PER_INTRINSIC] = unsafe { transmute(dest) };
            dest_cast[..intercept_chunk.len()].copy_from_slice(intercept_chunk);
        }

        // copy coefficients
        for chunk_num in 0..num_col_instrinsics {
            let mut col: Vec<__m256> = Vec::new();
            for r in rows {
                let chunk = r.chunks(SINGLES_PER_INTRINSIC).nth(chunk_num)?;
                let mut intrin = [0f32; SINGLES_PER_INTRINSIC];
                intrin[..chunk.len()].copy_from_slice(chunk);
                col.push(unsafe { transmute(intrin) });
            }
            mat.column_intrinsics.push(col);
        }

        Some(mat)
    }

    #[inline(always)]
    fn multiply_add(accumulate: &mut __m256, v1: __m256, v2: f32) {
        unsafe {
            // broadcast value (since we already have a reference)
            let val_broad = _mm256_set1_ps(v2);
            // separate multiply add is faster here
            let mult = _mm256_mul_ps(val_broad, v1);
            *accumulate = _mm256_add_ps(*accumulate, mult);
            // * not using FMA; it's slower here
            //accumulate = _mm256_fmadd_ps(val_broad, *row_intrin, accumulate);
        }
    }

    pub fn product(&self, values: &[f32], destination: &mut [f32]) -> Option<()> {
        if destination.len() != self.num_columns || values.len() != self.num_rows {
            return None;
        }

        destination
            .chunks_mut(SINGLES_PER_INTRINSIC)
            .zip(self.column_intrinsics.iter())
            .zip(self.intercept_intrinsics.iter())
            .for_each(|((dst, col), intercepts)| {
                // run multiplication and add to `accumulate`
                let mut accumulate = *intercepts;
                for (val, row_intrin) in values.iter().zip(col) {
                    Self::multiply_add(&mut accumulate, *row_intrin, *val);
                }
                // copy to destination (by interpreting the intrinsic as a slice) -- and we might
                // have a shorter final slice
                let src: &[f32; SINGLES_PER_INTRINSIC] = unsafe { transmute(&accumulate) };
                dst.copy_from_slice(&src[0..dst.len()]);
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
            .chunks_mut(SINGLES_PER_INTRINSIC)
            .zip(self.column_intrinsics.iter())
            .zip(self.intercept_intrinsics.iter())
            .for_each(|((dst, col), intercepts)| {
                // run multiplication and add to `accumulate`, starting with the intercepts
                let mut accumulate = *intercepts;
                for (val, row_intrin) in values.iter().zip(col) {
                    Self::multiply_add(&mut accumulate, *row_intrin, *val);
                }

                // copy to destination (taking into account final shorter stub) and apply cumulative softmax
                // 1. approximate exponential
                accumulate = crate::exp_approx_avx::exp_approx_avxf32(accumulate);
                // 2. accumulate and copy
                let src: &[f32; SINGLES_PER_INTRINSIC] = unsafe { transmute(&accumulate) };
                dst.iter_mut().zip(src).for_each(|(d, s)| {
                    cumulative_sum += s;
                    *d = cumulative_sum;
                });
            });

        Some(())
    }
}
