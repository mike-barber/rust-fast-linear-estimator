use std::arch::x86_64::*;

#[allow(non_camel_case_types)]
type vfloat = __m256;
#[allow(non_camel_case_types)]
type vint2 = __m256i;
#[allow(non_camel_case_types)]
type vmask = __m256i;
#[allow(non_camel_case_types)]
type vopmask = __m256i;

#[allow(non_upper_case_globals)]
const R_LN2f: f32 = 1.442695040888963407359924681001892137426645954152985934135449406931_f32;

#[allow(non_upper_case_globals)]
const L2Uf: f32 = 0.693145751953125_f32;

#[allow(non_upper_case_globals)]
const L2Lf: f32 = 1.428606765330187045e-06_f32;

#[allow(non_upper_case_globals)]
const SLEEF_INFINITYf: f32 = f32::INFINITY;

pub fn exp(x_in: __m256) -> __m256 {
    unsafe { xexpf(x_in) }
}

unsafe fn xexpf(d: vfloat) -> vfloat {
    let q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2f)));
    let mut s: vfloat;
    let mut u: vfloat;

    s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf), d);
    s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf), s);

    u = vcast_vf_f(0.000198527617612853646278381);
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219));
    u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5));

    u = vadd_vf_vf_vf(
        vcast_vf_f(1.0_f32),
        vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, s),
    );

    u = vldexp2_vf_vf_vi2(u, q);

    u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(
        vlt_vo_vf_vf(d, vcast_vf_f(-104.0)),
        vreinterpret_vm_vf(u),
    ));
    u = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(vcast_vf_f(100.0), d),
        vcast_vf_f(SLEEF_INFINITYf),
        u,
    );

    return u;
}

// static INLINE vint2 vcast_vi2_vm(vmask vm) { return vm; }
unsafe fn vrint_vi2_vf(vf: vfloat) -> vint2 {
    vcast_vi2_vm(_mm256_cvtps_epi32(vf))
}

// static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }
unsafe fn vcast_vi2_vm(vm: vmask) -> vint2 {
    return vm;
}

// static INLINE vmask vcast_vm_vi2(vint2 vi) { return vi; }
unsafe fn vcast_vm_vi2(vi: vint2) -> vmask {
    return vi;
}

// static INLINE vfloat vcast_vf_f(float f) { return _mm256_set1_ps(f); }
unsafe fn vcast_vf_f(f: f32) -> vfloat {
    return _mm256_set1_ps(f);
}

// static INLINE vfloat vcast_vf_vi2(vint2 vi) { return _mm256_cvtepi32_ps(vcast_vm_vi2(vi)); }
unsafe fn vcast_vf_vi2(vi: vint2) -> vfloat {
    return _mm256_cvtepi32_ps(vcast_vm_vi2(vi));
}

// static INLINE vint2 vcast_vi2_i(int i) { return _mm256_set1_epi32(i); }
unsafe fn vcast_vi2_i(i: i32) -> vint2 {
    return _mm256_set1_epi32(i);
}

//static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return _mm256_castsi256_ps(vm); }
unsafe fn vreinterpret_vf_vm(vm: vmask) -> vfloat {
    return _mm256_castsi256_ps(vm);
}

// static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return _mm256_castps_si256(vf); }
unsafe fn vreinterpret_vm_vf(vf: vfloat) -> vmask {
    return _mm256_castps_si256(vf);
}

// static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LT_OQ)); }
unsafe fn vlt_vo_vf_vf(x: vfloat, y: vfloat) -> vopmask {
    return vreinterpret_vm_vf(_mm256_cmp_ps(x, y, _CMP_LT_OQ));
}

// static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return _mm256_add_ps(x, y); }
unsafe fn vadd_vf_vf_vf(x: vfloat, y: vfloat) -> vfloat {
    return _mm256_add_ps(x, y);
}

// // static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return _mm256_sub_ps(x, y); }
// unsafe fn vsub_vf_vf_vf(x: vfloat, y: vfloat) -> vfloat {
//     return _mm256_sub_ps(x, y);
// }

// static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return _mm256_mul_ps(x, y); }
unsafe fn vmul_vf_vf_vf(x: vfloat, y: vfloat) -> vfloat {
    return _mm256_mul_ps(x, y);
}

// // static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return _mm256_div_ps(x, y); }
// unsafe fn vdiv_vf_vf_vf(x: vfloat, y: vfloat) -> vfloat {
//     return _mm256_div_ps(x, y);
// }

// static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return _mm256_fmadd_ps(x, y, z); }
unsafe fn vmla_vf_vf_vf_vf(x: vfloat, y: vfloat, z: vfloat) -> vfloat {
    return _mm256_fmadd_ps(x, y, z);
}

//static INLINE vint2 vsll_vi2_vi2_i(vint2 x, int c) { return _mm256_slli_epi32(x, c); }
unsafe fn vsll_vi2_vi2_i<const C: i32>(x: vint2) -> vint2 {
    return _mm256_slli_epi32(x, C);
}

// //static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, int c) { return _mm256_srli_epi32(x, c); }
// unsafe fn vsrl_vi2_vi2_i<const C: i32>(x: vint2) -> vint2 {
//     return _mm256_srli_epi32(x, C);
// }

//static INLINE vint2 vsra_vi2_vi2_i(vint2 x, int c) { return _mm256_srai_epi32(x, c); }
unsafe fn vsra_vi2_vi2_i<const C: i32>(x: vint2) -> vint2 {
    return _mm256_srai_epi32(x, C);
}

// static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) { return vreinterpret_vm_vd(_mm256_andnot_pd(vreinterpret_vd_vm(x), vreinterpret_vd_vm(y))); }
unsafe fn vandnot_vm_vo32_vm(x: vopmask, y: vmask) -> vmask {
    return _mm256_andnot_si256(x, y);
}

// static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) { return _mm256_blendv_ps(y, x, _mm256_castsi256_ps(o)); }
unsafe fn vsel_vf_vo_vf_vf(o: vopmask, x: vfloat, y: vfloat) -> vfloat {
    return _mm256_blendv_ps(y, x, _mm256_castsi256_ps(o));
}

// static INLINE CONST vfloat vldexp2_vf_vf_vi2(vfloat d, vint2 e) {
//     return vmul_vf_vf_vf(vmul_vf_vf_vf(d, vpow2i_vf_vi2(vsra_vi2_vi2_i(e, 1))), vpow2i_vf_vi2(vsub_vi2_vi2_vi2(e, vsra_vi2_vi2_i(e, 1))));
// }
unsafe fn vldexp2_vf_vf_vi2(d: vfloat, e: vint2) -> vfloat {
    return vmul_vf_vf_vf(
        vmul_vf_vf_vf(d, vpow2i_vf_vi2(vsra_vi2_vi2_i::<1>(e))),
        vpow2i_vf_vi2(vsub_vi2_vi2_vi2(e, vsra_vi2_vi2_i::<1>(e))),
    );
}

// static INLINE CONST vfloat vpow2i_vf_vi2(vint2 q) {
//     return vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f)), 23)));
// }
unsafe fn vpow2i_vf_vi2(q: vint2) -> vfloat {
    return vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i::<23>(vadd_vi2_vi2_vi2(
        q,
        vcast_vi2_i(0x7f),
    ))));
}

//static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_add_epi32(x, y); }
unsafe fn vadd_vi2_vi2_vi2(x: vint2, y: vint2) -> vint2 {
    return _mm256_add_epi32(x, y);
}
//static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return _mm256_sub_epi32(x, y); }
unsafe fn vsub_vi2_vi2_vi2(x: vint2, y: vint2) -> vint2 {
    return _mm256_sub_epi32(x, y);
}
