# TODO

- math_avxfun attribution: http://software-lisc.fbk.eu/avx_mathfun/avx_mathfun.h (zlib license)
- inavec attribution: https://gitlab.mpcdf.mpg.de/bbramas/inastemp 
    - constants taken from https://gitlab.mpcdf.mpg.de/bbramas/inastemp/-/blob/master/Src/Common/InaFastExp.hpp
    - as explained here: http://berenger.eu/blog/csimd-fast-exponential-computation-on-simd-architectures-implementation/
    - Remez approach is more accurate across the range than doing a least squares fit in of the polynomial in R with lm(...)
