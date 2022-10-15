#[cfg(test)]
mod tests {
    use std::iter::successors;

    use approx::abs_diff_eq;
    use approx::assert_abs_diff_eq;

    #[cfg(target_arch = "x86_64")]
    use crate::matrix_avx::MatrixF32;
    #[cfg(target_arch = "x86_64")]
    use crate::matrix_avx::SINGLES_PER_INTRINSIC;

    #[cfg(target_arch = "aarch64")]
    use crate::matrix_arm::MatrixF32;
    #[cfg(target_arch = "aarch64")]
    use crate::matrix_arm::SINGLES_PER_INTRINSIC;

    #[test]
    fn structure_create_exact() {
        // 5 rows, 10 columns
        let rows = vec![
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.],
        ];
        let intecepts = [1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.];

        let matrix = MatrixF32::create_from_rows(&rows, &intecepts).unwrap();

        let expected_col_intrinsics = 10 / SINGLES_PER_INTRINSIC
            + match 10 % SINGLES_PER_INTRINSIC {
                0 => 0,
                _ => 1,
            };

        assert_eq!(matrix.num_columns, 10);
        assert_eq!(matrix.num_col_instrinsics, expected_col_intrinsics);
        assert_eq!(matrix.num_rows, 5);
    }

    #[test]
    fn product() {
        // in R,
        //      > x = 1:2
        //      > coeff = t(matrix(1:6, ncol=2))
        //      > intercept = c(10,20,30)
        //      > x %*% coeff + intercept
        //           [,1] [,2] [,3]
        //      [1,]   19   32   45
        //
        let rows = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];
        let intercepts = [10f32, 20f32, 30f32];
        let matrix = MatrixF32::create_from_rows(&rows, &intercepts).unwrap();
        let v = vec![1f32, 2.];
        let mut res = vec![0f32; 3];
        matrix.product(&v, &mut res);
        assert_eq!(res, [19f32, 32., 45.]);
    }

    #[test]
    fn product_large() {
        let coeffs: Vec<f32> = (1..=(35 * 5)).map(|x| x as f32).collect();
        let rows: Vec<Vec<f32>> = coeffs[..].chunks(35).map(|c| c.to_vec()).collect();
        let intercepts = [0f32; 35]; // leave these zero; another test covers this

        let matrix = MatrixF32::create_from_rows(&rows, &intercepts).unwrap();
        let v: Vec<f32> = (1..=5).map(|x| x as f32).collect();

        // output to f32
        let mut res = vec![0f32; 35];
        matrix.product(&v, &mut res);
        assert_eq!(res[0], 1415_f32);
        assert_eq!(res[18], 1685_f32);
        assert_eq!(res[34], 1925_f32);
    }

    #[test]
    fn product_softmax() {
        // in R,
        //      > coeff = t(matrix(1:6, ncol=2))
        //      > intercept = c(0.1, 0.2, 0.3)
        //      > x = c(0.1, 0.5)
        //      > logit = x %*% coeff + intercept
        //      > cumsum(exp(logit))
        //      [1]  9.025013 27.199159 63.797393
        //
        let rows = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];
        let intercepts = [0.1f32, 0.2f32, 0.3f32];
        let matrix = MatrixF32::create_from_rows(&rows, &intercepts).unwrap();
        let v = vec![0.1f32, 0.5f32];

        // approx
        let mut res = vec![0f32; 3];
        matrix.product_softmax_cumulative_approx(&v, &mut res);

        // check approximately equal (with faily large tolerance since the numbers are large)
        let ok = res
            .iter()
            .zip(&[9.025013_f32, 27.199159_f32, 63.797393_f32])
            .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 0.01f32));
        assert!(ok);

        // sleef
        let mut res = vec![0f32; 3];
        matrix.product_softmax_cumulative_sleef(&v, &mut res);

        // check approximately equal (with faily large tolerance since the numbers are large)
        let ok = res
            .iter()
            .zip(&[9.025013_f32, 27.199159_f32, 63.797393_f32])
            .all(|(a, b)| abs_diff_eq!(a, b, epsilon = 0.01f32));
        assert!(ok);
    }

    #[test]
    fn product_softmax_not_normalised() {
        // in R,
        //      > coeff = t(matrix(1:40 / 100, ncol=2))
        //      > intercept = 1:20 / 100
        //      > x = c(-0.02, 0.04)
        //      > logit = x %*% coeff + intercept
        //      > exp(logit)
        //      [1,] 9.025013 18.17415 36.59823
        //      > sum(exp(logit))
        //      [1] 63.79739
        //
        let mut riter = successors(Some(0.01_f32), |n| Some(n + 0.01));
        let rows: Vec<Vec<f32>> = vec![
            (&mut riter).take(20).collect(),
            (&mut riter).take(20).collect(),
        ];
        let intercepts: Vec<f32> = std::iter::successors(Some(0.01_f32), |n| Some(n + 0.01))
            .take(20)
            .collect();
        let matrix = MatrixF32::create_from_rows(&rows, &intercepts).unwrap();
        let x = vec![-0.02, 0.04];

        let expected = vec![
            1.018367, 1.028807, 1.039355, 1.05001, 1.060775, 1.071651, 1.082637, 1.093737, 1.10495,
            1.116278, 1.127722, 1.139284, 1.150964, 1.162764, 1.174685, 1.186728, 1.198895,
            1.211186, 1.223603, 1.236148,
        ];
        let expected_sum = 22.47855;

        // approx with loose tolerance
        let tolerance: f32 = 1e-2;
        let mut res = vec![0f32; 20];
        let mut sum = 0.0;
        matrix.product_softmax_not_normalised_approx(&x, &mut res, &mut sum);

        // check approximately equal
        assert_abs_diff_eq!(expected_sum, sum, epsilon = tolerance);
        for (a, b) in res.iter().zip(&expected) {
            assert_abs_diff_eq!(a, b, epsilon = tolerance);
        }

        // sleef with tighter tolerance
        let tolerance: f32 = 1e-4;
        let mut res = vec![0f32; 20];
        let mut sum = 0.0;
        matrix.product_softmax_not_normalised_sleef(&x, &mut res, &mut sum);

        // check approximately equal (with faily large tolerance since the numbers are large)
        assert_abs_diff_eq!(expected_sum, sum, epsilon = tolerance);
        for (a, b) in res.iter().zip(&expected) {
            assert_abs_diff_eq!(a, b, epsilon = tolerance);
        }
    }
}
