use std::arch::x86_64::*;

use crate::exp_approx::exp_f32_const;

#[allow(dead_code)]
pub fn exp_approx_avxf32(x_in: __m256) -> __m256 {
    let mut x = x_in;
    unsafe {
        // clamp x
        x = _mm256_min_ps(x, _mm256_set1_ps(exp_f32_const::EXP_HI));
        x = _mm256_max_ps(x, _mm256_set1_ps(exp_f32_const::EXP_LO_AVX_SIGNED));

        // apply approximation
        x = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E));
        let fl = _mm256_floor_ps(x);
        let xf = _mm256_sub_ps(x, fl);

        let mut kn = _mm256_set1_ps(exp_f32_const::C3);
        // multiply add (no benefit from using FMA here, unfortunately)
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C2));
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C1));
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C0));
        x = _mm256_sub_ps(x, kn);

        // create integer with bits in the right place, by rounding double to integer,
        // then re-interpret as a double; again no benefit from using FMA here
        let xf32 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(exp_f32_const::S), x),
            _mm256_set1_ps(exp_f32_const::B),
        );
        let xul = _mm256_cvtps_epi32(xf32); // convert (numerically) to i32
        let res = _mm256_castsi256_ps(xul); // now cast back to f32

        res
    }
}

