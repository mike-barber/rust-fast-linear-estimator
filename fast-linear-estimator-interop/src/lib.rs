#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

mod direct {
    use std::slice;

    // coefficients: rows correspond to outputs; columns correspond to inputs;
    // ordering is ROW major (i.e. linear in row, stride for columns)
    #[no_mangle]
    pub extern "C" fn matrix_multiply_f64_row_major(
        num_inputs: usize,
        num_outputs: usize,
        coefficients: *const f64,
        inputs: *const f64,
        outputs: *mut f64,
    ) {
        // get slices
        let cf = unsafe { slice::from_raw_parts(coefficients, num_outputs * num_inputs) };
        let xx = unsafe { slice::from_raw_parts(inputs, num_inputs) };
        let yy = unsafe { slice::from_raw_parts_mut(outputs, num_outputs) };

        // zero y (assume it's not initialised)
        // not in stable yet: `yy.fill(0.);`
        yy.iter_mut().for_each(|y| *y = 0.);

        // then perform matrix multiply, adding to y for each input/row
        cf.chunks(num_inputs).zip(yy).for_each(|(cf_row, y)| {
            //println!("Row {:?}; y {:?}", cf_row, y);
            xx.iter().zip(cf_row.iter()).for_each(|(x, c)| {
                *y += x * c;
            });
        });
    }

    // coefficients: columns correspond to outputs; rows correspond to inputs;
    // ordering is COLUMN major (i.e. linear in column, stride for rows)
    #[no_mangle]
    pub extern "C" fn matrix_multiply_f64_column_major(
        num_inputs: usize,
        num_outputs: usize,
        coefficients: *const f64,
        inputs: *const f64,
        outputs: *mut f64,
    ) {
        // get slices
        let cf = unsafe { slice::from_raw_parts(coefficients, num_outputs * num_inputs) };
        let xx = unsafe { slice::from_raw_parts(inputs, num_inputs) };
        let yy = unsafe { slice::from_raw_parts_mut(outputs, num_outputs) };

        // zero y (assume it's not initialised)
        // not in stable yet: `yy.fill(0.);`
        yy.iter_mut().for_each(|y| *y = 0.);

        cf.chunks(num_outputs).zip(xx).for_each(|(cf_row, x)| {
            // dbg!(&cf_row);
            // dbg!(&x);
            yy.iter_mut().zip(cf_row.iter()).for_each(|(y, c)| {
                *y += x * c;
            });
            // dbg!(&yy);
        })
    }

    // same as above but no op; just for testing call cost
    #[no_mangle]
    pub extern "C" fn matrix_multiply_f64_row_major_no_op(
        _num_inputs: usize,
        _num_outputs: usize,
        _coefficients: *const f64,
        _inputs: *const f64,
        _outputs: *mut f64,
    ) {
        // no-op
    }

    #[cfg(test)]
    mod tests {

        #[test]
        fn add_works() {
            assert_eq!(5, crate::add(2, 3))
        }

        #[test]
        fn matrix_multiply_f64_row_major() {
            let coefficients = [[1., 4.], [2., 5.], [3., 6.]];
            let inputs = [1., 2.];
            let mut outputs = [0f64; 3];

            super::matrix_multiply_f64_row_major(
                2,
                3,
                coefficients[0].as_ptr(),
                inputs.as_ptr(),
                outputs.as_mut_ptr(),
            );

            assert_eq!(outputs, [9., 12., 15.]);
        }

        #[test]
        fn matrix_multiply_f64_column_major() {
            let coefficients = [[1., 2., 3.], [4., 5., 6.]];
            let inputs = [1., 2.];
            let mut outputs = [0f64; 3];

            super::matrix_multiply_f64_column_major(
                2,
                3,
                coefficients[0].as_ptr(),
                inputs.as_ptr(),
                outputs.as_mut_ptr(),
            );

            assert_eq!(outputs, [9., 12., 15.]);
        }
    }
}

mod avx_column_major {
    use fast_linear_estimator::matrix::MatrixAvxF32;
    use std::slice;

    // coefficients: columns correspond to outputs; rows correspond to inputs;
    // ordering is COLUMN major (i.e. linear in column, stride for rows)
    #[no_mangle]
    pub extern "C" fn create_avx_f32_matrix_column_major(
        num_inputs: usize,
        num_outputs: usize,
        coefficients: *const f32,
        intercepts: *const f32,
    ) -> *mut MatrixAvxF32 {
        let cf = unsafe { slice::from_raw_parts(coefficients, num_outputs * num_inputs) };
        let rows: Vec<Vec<f32>> = cf.chunks(num_outputs).map(|row| row.to_vec()).collect();

        let intercepts = unsafe { slice::from_raw_parts(intercepts, num_outputs) };

        if let Some(matrix) = MatrixAvxF32::create_from_rows(&rows, intercepts) {
            Box::into_raw(Box::new(matrix))
        } else {
            std::ptr::null_mut()
        }
    }

    #[no_mangle]
    pub extern "C" fn product_avx_f32_matrix_column_major(
        matrix: *mut MatrixAvxF32,
        values: *const f32,
        values_length: usize,
        results: *mut f32,
        results_length: usize,
    ) -> bool {
        // get reference to the matrix, but don't take ownership of it
        if matrix.is_null() {
            return false;
        }
        let mat = unsafe { Box::leak(Box::from_raw(matrix)) };

        // check input and result pointers
        if values.is_null() || results.is_null() {
            return false;
        }

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

    // clean up matrix
    #[no_mangle]
    pub unsafe extern "C" fn destroy_avx_f32_matrix_column_major(matrix: *mut MatrixAvxF32) {
        if !matrix.is_null() {
            drop(Box::from_raw(matrix));
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
                super::create_avx_f32_matrix_column_major(2, 3, 
                    coefficients[0].as_ptr(),
                    intercepts.as_ptr()
                );

            // test using f32 result
            assert!(super::product_avx_f32_matrix_column_major(
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
            assert!(super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert_eq!(results, [109_f32, 212_f32, 315_f32]);

            // wrong input size
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                1, // wrong
                results.as_mut_ptr(),
                3
            ));

            // wrong result size
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2, 
                results.as_mut_ptr(),
                1 // wrong
            ));

            // null input
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                std::ptr::null(),
                2, 
                results.as_mut_ptr(),
                3 
            ));

            // null result
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2, 
                std::ptr::null_mut(),
                3 
            ));

            // now destroy the matrix -- do this only once (caller's responsibility)
            unsafe { super::destroy_avx_f32_matrix_column_major(matrix) };
            matrix = std::ptr::null_mut();

            // and finally attempting to use it should fail with false (not panic)
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
            assert!(!super::product_avx_f32_matrix_column_major(
                matrix,
                inputs.as_ptr(),
                2,
                results.as_mut_ptr(),
                3
            ));
        }
    }
}
