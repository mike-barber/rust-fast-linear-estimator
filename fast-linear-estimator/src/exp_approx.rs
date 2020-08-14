use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_cvtss_f32,
    _mm256_floor_ps, _mm256_max_ps, _mm256_min_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_sub_ps,
};

const EXP_BIAS_32: i32 = 127; // zero point for exponent

mod exp_f32_const {
    // taken from cephes/avxfun
    pub const EXP_HI: f32 = 88.3762626647949;
    pub const EXP_LO: f32 = -88.3762626647949;
    // reduced negative limit because we're converting to signed integers
    pub const EXP_LO_AVX_SIGNED: f32 = -88.028;

    // taken from inavec
    pub const C0: f32 = 1.06906116358144185133e-04;
    pub const C1: f32 = 3.03543677780836240743e-01;
    pub const C2: f32 = -2.24339532327269441936e-01;
    pub const C3: f32 = -7.92041454535668681958e-02;

    pub const S: f32 = (1u64 << 23) as f32;
    pub const B: f32 = S * (super::EXP_BIAS_32 as f32);
}

#[allow(dead_code)]
pub fn exp_approx_f32(x_in: f32) -> f32 {
    // clamp x
    let mut x = x_in;
    x = x.min(exp_f32_const::EXP_HI);
    x = x.max(exp_f32_const::EXP_LO);

    // apply approximation
    x = x * std::f32::consts::LOG2_E;
    let fl = x.floor();
    let xf = x - fl;

    let mut kn = exp_f32_const::C3;
    kn = xf * kn + exp_f32_const::C2;
    kn = xf * kn + exp_f32_const::C1;
    kn = xf * kn + exp_f32_const::C0;
    x -= kn;

    // create integer with bits in the right place, by rounding double to integer,
    // then re-interpret as a double
    let xul = ((exp_f32_const::S * x) + exp_f32_const::B) as u32;
    let res = unsafe { std::mem::transmute::<u32, f32>(xul) };
    res
}

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
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C2));
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C1));
        kn = _mm256_add_ps(_mm256_mul_ps(xf, kn), _mm256_set1_ps(exp_f32_const::C0));
        x = _mm256_sub_ps(x, kn);

        // create integer with bits in the right place, by rounding double to integer,
        // then re-interpret as a double
        let xf32 = _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(exp_f32_const::S), x),
            _mm256_set1_ps(exp_f32_const::B),
        );
        let xul = _mm256_cvtps_epi32(xf32); // convert (numerically) to i32
        let res = _mm256_castsi256_ps(xul); // now cast back to f32

        res
    }
}

#[allow(dead_code)]
pub fn avx_ps_first_element(v: __m256) -> f32 {
    unsafe { _mm256_cvtss_f32(v) }
}

#[cfg(test)]
mod tests {

    use approx::*;
    use std::arch::x86_64::*;

    const VALS: [f32; 8] = [-10_f32, -5., -1., 0., 1., 2., 5., 10.];

    fn expected() -> Vec<f32> {
        VALS.iter().map(|v| v.exp()).collect()
    }

    fn check_assert(vals: &[f32]) {
        vals.iter().zip(expected().iter()).for_each(|(act, exp)| {
            assert_relative_eq!(exp, act, max_relative = 1e-4);
        });
    }

    #[test]
    fn inavec_exp_approx_f32() {
        let res: Vec<_> = VALS
            .iter()
            .map(|&v| super::exp_approx_f32(v))
            .collect();
        check_assert(&res);
    }

    #[test]
    fn inavec_exp_approx_avxf32() {
        let input: __m256 = unsafe { _mm256_loadu_ps(&VALS[0]) };
        let res = super::exp_approx_avxf32(input);

        let res_f32 = crate::common::m256_f32_to_vec(&[res]);
        check_assert(&res_f32);
    }
}
