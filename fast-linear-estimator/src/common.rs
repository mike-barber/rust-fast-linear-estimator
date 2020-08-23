#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256, _mm256_setzero_ps};
use std::{mem::transmute_copy, slice};

// return zero avx single
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub fn m256_f32_zero() -> __m256 {
    unsafe { _mm256_setzero_ps() }
}

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub fn m256_f32_to_vec(vals: &[__m256]) -> Vec<f32> {
    let res: Vec<f32> = vals
        .iter()
        .flat_map(|v| {
            let vec = unsafe { transmute_copy::<__m256, [f32; 8]>(v) }.to_vec();
            vec
        })
        .collect();
    res
}

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub fn m256_f32_to_existing_slice(vals: &[__m256], dest: &mut [f32]) {
    unsafe {
        let elements = slice::from_raw_parts(vals.as_ptr() as *const f32, vals.len() * 8);
        dest.copy_from_slice(&elements[0..dest.len()]);
    }
}
