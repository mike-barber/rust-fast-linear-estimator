const EXP_BIAS_32: i32 = 127; // zero point for exponent

pub mod exp_f32_const {
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
pub fn exp_approx_slice_in_place(vals: &mut [f32]) {
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps};
    use crate::exp_approx_avx::exp_approx_avxf32;

    let mut chunks = vals.chunks_exact_mut(8);
    (&mut chunks).for_each(|ch| {
        let mut v = unsafe { _mm256_loadu_ps(ch.as_ptr()) };
        v = exp_approx_avxf32(v);
        unsafe {
            _mm256_storeu_ps(ch.as_mut_ptr(), v);
        }
    });

    let rem = chunks.into_remainder();
    if rem.len() > 0 {
        unsafe {
            let mut v = _mm256_setzero_ps();
            
            let slice_in: &mut [f32;8] = std::mem::transmute(&mut v);
            slice_in[0..rem.len()].copy_from_slice(rem);

            let res = exp_approx_avxf32(v);
            
            let slice_out: &[f32;8] = std::mem::transmute(&res);
            rem.copy_from_slice(&slice_out[0..rem.len()]);
        }
    }
}

#[allow(dead_code)]
#[cfg(target_arch = "aarch64")]
fn exp_approx_slice_in_place(vals: &mut [f32]) {
    todo!();
}

#[cfg(test)]
mod tests {

    use approx::*;

    const VALS: [f32; 8] = [-10_f32, -5., -1., 0., 1., 2., 5., 10.];

    fn expected(vals: &[f32]) -> Vec<f32> {
        vals.iter().map(|v| v.exp()).collect()
    }

    fn check_assert(expect: &[f32], vals: &[f32]) {
        vals.iter().zip(expect.iter()).for_each(|(act, exp)| {
            assert_relative_eq!(exp, act, max_relative = 1e-4);
        });
    }

    #[test]
    fn exp_approx_f32() {
        let res: Vec<_> = VALS.iter().map(|&v| super::exp_approx_f32(v)).collect();
        let expect = expected(&VALS);
        check_assert(&expect, &res);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn exp_approx_avxf32() {
        use std::arch::x86_64::*;

        let input: __m256 = unsafe { _mm256_loadu_ps(&VALS[0]) };
        let res = crate::exp_approx_avx::exp_approx_avxf32(input);

        let res_f32: [f32; 8] = unsafe { std::mem::transmute(res) };
        check_assert(&expected(&VALS), &res_f32);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn exp_approx_armf32() {
        use std::arch::aarch64::*;

        unsafe {
            let x1: float32x4_t = std::mem::transmute([-10_f32, -5., -1., 0.]);
            let x2: float32x4_t = std::mem::transmute([1.0_f32, 2., 5., 10.]);

            let res1: [f32; 4] = std::mem::transmute(crate::exp_approx_arm::exp_approx_armf32(x1));
            let res2: [f32; 4] = std::mem::transmute(crate::exp_approx_arm::exp_approx_armf32(x2));

            let expect = expected();
            check_assert(&expect[0..4], &res1);
            check_assert(&expect[4..8], &res2);
        }
    }

    #[test]
    fn exp_approx_slice_in_place() {
        let mut vals = vec![-3f32,-2.,-1.,0.,1.,2.,3.,4.,5.,6.,7.,8.,9.];
        let expected = expected(&vals);

        crate::exp_approx::exp_approx_slice_in_place(&mut vals);
        
        check_assert(&expected, &vals);
    }
}
