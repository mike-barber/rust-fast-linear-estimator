use std::arch::aarch64::*;

use crate::exp_approx::exp_f32_const;

#[allow(dead_code)]
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

