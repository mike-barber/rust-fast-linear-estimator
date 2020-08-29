#![cfg_attr(feature = "nightly", feature(stdsimd, asm))]


//
// exponential approximation
//
pub mod exp_approx;

#[cfg(target_arch = "x86_64")]
pub mod exp_approx_avx;

#[cfg(target_arch = "aarch64")]
pub mod exp_approx_arm;

//
// matrix implementation
//
#[cfg(target_arch = "x86_64")]
pub mod matrix_avx;

#[cfg(target_arch = "aarch64")]
pub mod matrix_arm;

pub mod matrix_test;

