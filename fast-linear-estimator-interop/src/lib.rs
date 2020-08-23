#[no_mangle]
pub extern "C" fn test_add(a: i32, b: i32) -> i32 {
    a + b
}

mod avx_f32 {
    use fast_linear_estimator::matrix::MatrixF32;
    use std::slice;

    // coefficients: columns correspond to outputs; rows correspond to inputs;
    // ordering is COLUMN major (i.e. linear in column, stride for rows)
    #[no_mangle]
    pub extern "C" fn matrix_f32_create(
        num_inputs: usize,
        num_outputs: usize,
        coefficients: *const f32,
        intercepts: *const f32,
    ) -> *mut MatrixF32 {
        let cf = unsafe { slice::from_raw_parts(coefficients, num_outputs * num_inputs) };
        let rows: Vec<Vec<f32>> = cf.chunks(num_outputs).map(|row| row.to_vec()).collect();

        let intercepts = unsafe { slice::from_raw_parts(intercepts, num_outputs) };

        if let Some(matrix) = MatrixF32::create_from_rows(&rows, intercepts) {
            Box::into_raw(Box::new(matrix))
        } else {
            std::ptr::null_mut()
        }
    }

    // clean up matrix
    #[no_mangle]
    pub unsafe extern "C" fn matrix_avx_f32_delete(matrix: *mut MatrixF32) {
        if !matrix.is_null() {
            drop(Box::from_raw(matrix));
        }
    }

    #[no_mangle]
    pub extern "C" fn matrix_f32_product(
        matrix: *mut MatrixF32,
        values: *const f32,
        values_length: usize,
        results: *mut f32,
        results_length: usize,
    ) -> bool {
        // check for nulls
        if matrix.is_null() || values.is_null() || results.is_null() {
            return false;
        }
        // get reference to the matrix, but don't take ownership of it
        let mat = unsafe { Box::leak(Box::from_raw(matrix)) };
        // get slices for inputs and outputs
        let vals = unsafe { slice::from_raw_parts(values, values_length) };
        let res = unsafe { slice::from_raw_parts_mut(results, results_length) };

        // perform multiplication
        if let Some(()) = mat.product(vals, res) {
            true
        } else {
            false
        }
    }

    #[no_mangle]
    pub extern "C" fn matrix_f32_softmax_cumulative(
        matrix: *mut MatrixF32,
        values: *const f32,
        values_length: usize,
        results: *mut f32,
        results_length: usize,
    ) -> bool {
        // check for nulls
        if matrix.is_null() || values.is_null() || results.is_null() {
            return false;
        }
        // get reference to the matrix, but don't take ownership of it
        let mat = unsafe { Box::leak(Box::from_raw(matrix)) };
        // get slices for inputs and outputs
        let vals = unsafe { slice::from_raw_parts(values, values_length) };
        let res = unsafe { slice::from_raw_parts_mut(results, results_length) };

        // perform multiplication
        if let Some(()) = mat.product_softmax_cumulative_approx(vals, res) {
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    mod tests {

        // Testing native call interface, including safety after disposal

        #[test]
        fn avx_matrix_interface() {
            let coefficients = [[1_f32, 2., 3.], [4., 5., 6.]];
            let intercepts = [100_f32, 200_f32, 300_f32];
            let inputs = [1_f32, 2.];
            let mut results = [0_f32; 3];

            let mut matrix =
                super::matrix_f32_create(2, 3, coefficients[0].as_ptr(), intercepts.as_ptr());

            // test using f32 result
            assert!(super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert_eq!(results, [109_f32, 212_f32, 315_f32]);

            // put junk into results and test again
            // should get the same result, and the product function should not have freed the matrix
            results[0] = 1000.0;
            results[1] = 1000.0;
            results[2] = 1000.0;
            assert!(super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert_eq!(results, [109_f32, 212_f32, 315_f32]);

            // wrong input size
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                1, // wrong
                results.as_mut_ptr(),
                3
            ));

            // wrong result size
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                1 // wrong
            ));

            // null input
            assert!(!super::matrix_f32_product(
                matrix,
                std::ptr::null(),
                2,
                results.as_mut_ptr(),
                3
            ));

            // null result
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                std::ptr::null_mut(),
                3
            ));

            // now destroy the matrix -- do this only once (caller's responsibility)
            unsafe { super::matrix_avx_f32_delete(matrix) };
            matrix = std::ptr::null_mut();

            // and finally attempting to use it should fail with false (not panic)
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert!(!super::matrix_f32_product(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
        }
    }
}
