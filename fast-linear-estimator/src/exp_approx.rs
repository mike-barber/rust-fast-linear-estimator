#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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
#[cfg(target_arch = "x86_64")]
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

#[allow(dead_code)]
#[cfg(target_arch = "aarch64")]
pub fn exp_approx_armf32(x_in: float32x4_t) -> float32x4_t {
    use std::mem::{transmute, transmute_copy};

    const C_LOG2E: [f32; 4] = [
        std::f32::consts::LOG2_E,
        std::f32::consts::LOG2_E,
        std::f32::consts::LOG2_E,
        std::f32::consts::LOG2_E,
    ];
    const C_C3: [f32; 4] = [
        exp_f32_const::C3,
        exp_f32_const::C3,
        exp_f32_const::C3,
        exp_f32_const::C3,
    ];
    const C_C2: [f32; 4] = [
        exp_f32_const::C2,
        exp_f32_const::C2,
        exp_f32_const::C2,
        exp_f32_const::C2,
    ];
    const C_C1: [f32; 4] = [
        exp_f32_const::C1,
        exp_f32_const::C1,
        exp_f32_const::C1,
        exp_f32_const::C1,
    ];
    const C_C0: [f32; 4] = [
        exp_f32_const::C0,
        exp_f32_const::C0,
        exp_f32_const::C0,
        exp_f32_const::C0,
    ];
    const C_S: [f32; 4] = [
        exp_f32_const::S,
        exp_f32_const::S,
        exp_f32_const::S,
        exp_f32_const::S,
    ];
    const C_B: [f32; 4] = [
        exp_f32_const::B,
        exp_f32_const::B,
        exp_f32_const::B,
        exp_f32_const::B,
    ];

    let mut x = x_in;
    unsafe {
        // clamp x
        // note: can't seem to find the right instructions defined
        //       will rewrite later
        // x = vmin_f32(x, vld1q_dup_f32(&exp_f32_const::EXP_HI));
        // x = vmax_f32(x, vld1q_dup_f32(&exp_f32_const::EXP_LO_AVX_SIGNED));
        {
            let xv: &mut [f32; 4] = transmute(&mut x);
            xv[0] = xv[0]
                .min(exp_f32_const::EXP_HI)
                .max(exp_f32_const::EXP_LO_AVX_SIGNED);
            xv[1] = xv[1]
                .min(exp_f32_const::EXP_HI)
                .max(exp_f32_const::EXP_LO_AVX_SIGNED);
            xv[2] = xv[2]
                .min(exp_f32_const::EXP_HI)
                .max(exp_f32_const::EXP_LO_AVX_SIGNED);
            xv[3] = xv[3]
                .min(exp_f32_const::EXP_HI)
                .max(exp_f32_const::EXP_LO_AVX_SIGNED);
        }

        // apply approximation
        x = vmulq_f32(x, transmute(C_LOG2E));
        let fl: float32x4_t;
        {
            let xv: &mut [f32; 4] = transmute(&mut x);
            fl = transmute([xv[0].floor(), xv[1].floor(), xv[2].floor(), xv[3].floor()]);
        }
        let xf = vsubq_f32(x, fl);

        let mut kn: float32x4_t = transmute(C_C3);
        kn = vaddq_f32(vmulq_f32(xf, kn), transmute(C_C2));
        kn = vaddq_f32(vmulq_f32(xf, kn), transmute(C_C1));
        kn = vaddq_f32(vmulq_f32(xf, kn), transmute(C_C0));
        x = vsubq_f32(x, kn);

        // create integer with bits in the right place, by rounding double to integer,
        // then re-interpret as a double; again no benefit from using FMA here
        let xf32 = vaddq_f32(vmulq_f32(transmute(C_S), x), transmute(C_B));

        let xf32v: &[f32; 8] = transmute(&xf32);

        let i0: i32 = xf32v[0] as i32;
        let i1: i32 = xf32v[1] as i32;
        let i2: i32 = xf32v[2] as i32;
        let i3: i32 = xf32v[3] as i32;

        let f0: f32 = transmute(i0);
        let f1: f32 = transmute(i1);
        let f2: f32 = transmute(i2);
        let f3: f32 = transmute(i3);

        transmute_copy(&[f0, f1, f2, f3])
    }
}

#[cfg(test)]
mod tests {

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    use approx::*;

    const VALS: [f32; 8] = [-10_f32, -5., -1., 0., 1., 2., 5., 10.];

    fn expected() -> Vec<f32> {
        VALS.iter().map(|v| v.exp()).collect()
    }

    fn check_assert(expect: &[f32], vals: &[f32]) {
        vals.iter().zip(expect.iter()).for_each(|(act, exp)| {
            assert_relative_eq!(exp, act, max_relative = 1e-4);
        });
    }

    #[test]
    fn exp_approx_f32() {
        let res: Vec<_> = VALS.iter().map(|&v| super::exp_approx_f32(v)).collect();
        check_assert(&expected(), &res);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_approx_avxf32() {
        let input: __m256 = unsafe { _mm256_loadu_ps(&VALS[0]) };
        let res = super::exp_approx_avxf32(input);

        let res_f32 = crate::common::m256_f32_to_vec(&[res]);
        check_assert(&expected(), &res_f32);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_approx_armf32() {
        unsafe {
            let x1: float32x4_t = std::mem::transmute([-10_f32, -5., -1., 0.]);
            let x2: float32x4_t = std::mem::transmute([1.0_f32, 2., 5., 10.]);

            let res1:[f32;4] = std::mem::transmute(super::exp_approx_armf32(x1));
            let res2:[f32;4] = std::mem::transmute(super::exp_approx_armf32(x2));

            check_assert(&expected()[0..4], &res1);
            check_assert(&expected()[4..8], &res2);
        }
    }
}
