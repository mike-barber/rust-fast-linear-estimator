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
#[inline(always)]
pub fn exp_approx_armf32(x_in: float32x4_t) -> float32x4_t {
    use std::mem::transmute;

    unsafe {
        
        // multiply, clamp and calculate the fractional part
        let mut x = x_in;
        let mut xf: float32x4_t;
        asm!(
            "dup    {lov:v}.4s, {lo:w}",                    // broadcast to vector
            "dup    {hiv:v}.4s, {hi:w}",                    // broadcast to vector
            "fmax   {x:v}.4s,   {x:v}.4s,   {lov:v}.4s",    // max
            "fmin   {x:v}.4s,   {x:v}.4s,   {hiv:v}.4s",    // min
            "fmul   {x:v}.4s,   {x:v}.4s,   {l2e:v}.s[0]",  // multiply by scalar
            "frintm {fl:v}.4s,  {x:v}.4s",                  // floor
            "fsub   {xf:v}.4s,  {x:v}.4s,   {fl:v}.4s",    // xf = x - fl (fractional part)
            hi = in(reg) exp_f32_const::EXP_HI,
            lo = in(reg) exp_f32_const::EXP_LO_AVX_SIGNED,
            l2e = in(vreg) std::f32::consts::LOG2_E,
            hiv = out(vreg) _, // clobbered
            lov = out(vreg) _, // clobbered
            fl = out(vreg) _, // clobbered
            x = inout(vreg) x,
            xf = out(vreg) xf,
            options(nostack,nomem,pure)
        );

        // calculate the approximation
        asm!(
            "dup    {kn:v}.4s,  {c3:w}",                    // kn = c3
            
            "dup    {c:v}.4s,   {c2:w}",
            "fmul   {a:v}.4s,   {xf:v}.4s,  {kn:v}.4s",     // a = xf * kn
            "fadd   {kn:v}.4s,  {a:v}.4s,   {c:v}.4s",      // kn = a + c2

            "dup    {c:v}.4s,   {c1:w}",        
            "fmul   {a:v}.4s,   {xf:v}.4s,  {kn:v}.4s",     // a = xf * kn
            "fadd   {kn:v}.4s,  {a:v}.4s,   {c:v}.4s",      // kn = a + c1

            "dup    {c:v}.4s,   {c0:w}",
            "fmul   {a:v}.4s,   {xf:v}.4s,  {kn:v}.4s",     // a = xf * kn
            "fadd   {kn:v}.4s,  {a:v}.4s,   {c:v}.4s",      // kn = a + c0

            "fsub   {x:v}.4s,   {x:v}.4s,   {kn:v}.4s",     // x = x - kn
            c3 = in(reg) exp_f32_const::C3,
            c2 = in(reg) exp_f32_const::C2,
            c1 = in(reg) exp_f32_const::C1,
            c0 = in(reg) exp_f32_const::C0,
            xf = in(vreg) xf,
            kn = out(vreg) _,
            a = out(vreg) _,
            c = out(vreg) _,
            x = inout(vreg) x,
            options(nostack,nomem,pure)
        );

        // create integer with bits in the right place, by rounding double to integer,
        // then re-interpret as a double; again no benefit from using FMA here
        let xi: int32x4_t;
        asm!(
            "dup    {Bv:v}.4s,  {B:w}",                     // broadcast B
            "fmul   {x:v}.4s,   {x:v}.4s,   {S:v}.s[0]",    // x = x * S (with first element)
            "fadd   {x:v}.4s,   {x:v}.4s,   {Bv:v}.4s",     // x = x + B
            
            "fcvtzs {xi:v}.4s,  {x:v}.4s",                  // i = x as i32 (convert, not cast)
            S = in(vreg) exp_f32_const::S,                  // scalar within a 4-vector (first element)
            B = in(reg) exp_f32_const::B,
            x = in(vreg) x,
            xi = out(vreg) xi,
            Bv = out(vreg) _,
            options(pure,nomem,nostack)
        );
        // now cast and return
        transmute(xi)
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
        let expect = expected();
        check_assert(&expect, &res);
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

            let res1: [f32; 4] = std::mem::transmute(super::exp_approx_armf32(x1));
            let res2: [f32; 4] = std::mem::transmute(super::exp_approx_armf32(x2));

            let expect = expected();
            check_assert(&expect[0..4], &res1);
            check_assert(&expect[4..8], &res2);
        }
    }
}
