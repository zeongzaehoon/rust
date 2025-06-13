use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::f64;



#[pyfunction]
fn list_to_ndarray<'py>(py: Python<'py>, data: Vec<f64>) -> &'py PyArray1<f64> {
    let array = Array1::from(data);
    array.into_pyarray(py)
}

#[pyfunction]
fn flatten_to_matrix<'py>(
    py: Python<'py>, 
    data: PyReadonlyArray1<f64>, 
    rows: usize, 
    cols: usize
) -> PyResult<&'py PyArray2<f64>> {
    let array = data.as_array();
    
    if array.len() != rows * cols {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "데이터 크기가 행렬 크기와 맞지 않습니다."
        ));
    }
    
    let reshaped = array.to_owned().into_shape((rows, cols))
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "배열 형태 변환에 실패했습니다."
        ))?;
    
    Ok(reshaped.into_pyarray(py))
}

#[pyfunction]
fn normalize<'py>(py: Python<'py>, data: PyReadonlyArray1<f64>) -> &'py PyArray1<f64> {
    let array = data.as_array();
    
    if array.is_empty() {
        return Array1::from(vec![]).into_pyarray(py);
    }
    
    let min_val = array.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val + 1e-8;
    
    let normalized = array.mapv(|x| (x - min_val) / range);
    normalized.into_pyarray(py)
}

#[pyfunction]
fn standardize<'py>(py: Python<'py>, data: PyReadonlyArray1<f64>) -> &'py PyArray1<f64> {
    let array = data.as_array();
    
    if array.is_empty() {
        return Array1::from(vec![]).into_pyarray(py);
    }
    
    let mean = array.mean().unwrap_or(0.0);
    let std_dev = array.std(0.0) + 1e-8;
    
    let standardized = array.mapv(|x| (x - mean) / std_dev);
    standardized.into_pyarray(py)
}

#[pyfunction]
fn fillna_with_mean<'py>(py: Python<'py>, data: Vec<Option<f64>>) -> &'py PyArray1<f64> {
    let valid_values: Vec<f64> = data.iter().filter_map(|&x| x).collect();
    
    let mean = if valid_values.is_empty() {
        0.0
    } else {
        valid_values.iter().sum::<f64>() / valid_values.len() as f64
    };
    
    let filled: Vec<f64> = data.iter().map(|&x| x.unwrap_or(mean)).collect();
    Array1::from(filled).into_pyarray(py)
}

#[pyfunction]
fn filter_positive<'py>(py: Python<'py>, data: PyReadonlyArray1<f64>) -> &'py PyArray1<f64> {
    let array = data.as_array();
    let filtered: Vec<f64> = array.iter().filter(|&&x| x > 0.0).cloned().collect();
    Array1::from(filtered).into_pyarray(py)
}

#[pyfunction]
fn sort_by_column<'py>(
    py: Python<'py>, 
    matrix: PyReadonlyArray2<f64>, 
    col: usize
) -> PyResult<&'py PyArray2<f64>> {
    let array = matrix.as_array();
    
    if array.is_empty() {
        return Ok(Array2::zeros((0, 0)).into_pyarray(py));
    }
    
    if col >= array.ncols() {
        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
            "컬럼 인덱스가 범위를 벗어났습니다."
        ));
    }
    
    let mut rows_with_indices: Vec<(usize, ArrayView1<f64>)> = array
        .axis_iter(Axis(0))
        .enumerate()
        .collect();
    
    rows_with_indices.sort_by(|(_, a), (_, b)| {
        a[col].partial_cmp(&b[col]).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let sorted_data: Vec<Vec<f64>> = rows_with_indices
        .into_iter()
        .map(|(_, row)| row.to_vec())
        .collect();
    
    let rows = sorted_data.len();
    let cols = if rows > 0 { sorted_data[0].len() } else { 0 };
    let flat_data: Vec<f64> = sorted_data.into_iter().flatten().collect();
    
    let result = Array2::from_shape_vec((rows, cols), flat_data)
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "정렬된 배열 생성에 실패했습니다."
        ))?;
    
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn hstack_arrays<'py>(
    py: Python<'py>, 
    arrays: Vec<PyReadonlyArray2<f64>>
) -> PyResult<&'py PyArray2<f64>> {
    if arrays.is_empty() {
        return Ok(Array2::zeros((0, 0)).into_pyarray(py));
    }
    
    let first_array = arrays[0].as_array();
    let rows = first_array.nrows();
    
    // 모든 배열의 행 수가 같은지 확인
    for arr in &arrays {
        if arr.as_array().nrows() != rows {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "모든 배열의 행 수가 같아야 합니다."
            ));
        }
    }
    
    let total_cols: usize = arrays.iter().map(|arr| arr.as_array().ncols()).sum();
    let mut result = Array2::zeros((rows, total_cols));
    
    let mut col_offset = 0;
    for arr in arrays {
        let array = arr.as_array();
        let end_col = col_offset + array.ncols();
        result.slice_mut(s![.., col_offset..end_col]).assign(&array);
        col_offset = end_col;
    }
    
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn vstack_arrays<'py>(
    py: Python<'py>, 
    arrays: Vec<PyReadonlyArray2<f64>>
) -> PyResult<&'py PyArray2<f64>> {
    if arrays.is_empty() {
        return Ok(Array2::zeros((0, 0)).into_pyarray(py));
    }
    
    let first_array = arrays[0].as_array();
    let cols = first_array.ncols();
    
    // 모든 배열의 열 수가 같은지 확인
    for arr in &arrays {
        if arr.as_array().ncols() != cols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "모든 배열의 열 수가 같아야 합니다."
            ));
        }
    }
    
    let total_rows: usize = arrays.iter().map(|arr| arr.as_array().nrows()).sum();
    let mut result = Array2::zeros((total_rows, cols));
    
    let mut row_offset = 0;
    for arr in arrays {
        let array = arr.as_array();
        let end_row = row_offset + array.nrows();
        result.slice_mut(s![row_offset..end_row, ..]).assign(&array);
        row_offset = end_row;
    }
    
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn to_float32<'py>(py: Python<'py>, data: PyReadonlyArray1<f64>) -> &'py PyArray1<f32> {
    let array = data.as_array();
    let converted = array.mapv(|x| x as f32);
    converted.into_pyarray(py)
}

#[pyfunction]
fn expand_dims<'py>(
    py: Python<'py>, 
    data: PyReadonlyArray1<f64>, 
    axis: usize
) -> PyResult<&'py PyArray2<f64>> {
    let array = data.as_array();
    
    let result = match axis {
        0 => {
            // 행으로 추가: (n,) -> (1, n)
            array.to_owned().insert_axis(Axis(0))
        },
        1 => {
            // 열로 추가: (n,) -> (n, 1)
            array.to_owned().insert_axis(Axis(1))
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "axis는 0 또는 1이어야 합니다."
        ))
    };
    
    Ok(result.into_pyarray(py))
}

// 추가 벡터 연산 함수들
#[pyfunction]
fn vector_add<'py>(
    py: Python<'py>, 
    a: PyReadonlyArray1<f64>, 
    b: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let array_a = a.as_array();
    let array_b = b.as_array();
    
    if array_a.len() != array_b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "배열 크기가 같아야 합니다."
        ));
    }
    
    let result = &array_a + &array_b;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn vector_multiply<'py>(
    py: Python<'py>, 
    a: PyReadonlyArray1<f64>, 
    b: PyReadonlyArray1<f64>
) -> PyResult<&'py PyArray1<f64>> {
    let array_a = a.as_array();
    let array_b = b.as_array();
    
    if array_a.len() != array_b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "배열 크기가 같아야 합니다."
        ));
    }
    
    let result = &array_a * &array_b;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn scalar_multiply<'py>(
    py: Python<'py>, 
    data: PyReadonlyArray1<f64>, 
    scalar: f64
) -> &'py PyArray1<f64> {
    let array = data.as_array();
    let result = array.mapv(|x| x * scalar);
    result.into_pyarray(py)
}

#[pyfunction]
fn vector_sum(data: PyReadonlyArray1<f64>) -> f64 {
    data.as_array().sum()
}

#[pyfunction]
fn vector_mean(data: PyReadonlyArray1<f64>) -> Option<f64> {
    let array = data.as_array();
    if array.is_empty() {
        None
    } else {
        Some(array.mean().unwrap_or(0.0))
    }
}

#[pyfunction]
fn matrix_multiply<'py>(
    py: Python<'py>, 
    a: PyReadonlyArray2<f64>, 
    b: PyReadonlyArray2<f64>
) -> PyResult<&'py PyArray2<f64>> {
    let array_a = a.as_array();
    let array_b = b.as_array();
    
    if array_a.ncols() != array_b.nrows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "첫 번째 행렬의 열 수와 두 번째 행렬의 행 수가 같아야 합니다."
        ));
    }
    
    let result = array_a.dot(&array_b);
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn transpose<'py>(py: Python<'py>, matrix: PyReadonlyArray2<f64>) -> &'py PyArray2<f64> {
    let array = matrix.as_array();
    let transposed = array.t().to_owned();
    transposed.into_pyarray(py)
}

#[pyfunction]
fn array_max(data: PyReadonlyArray1<f64>) -> Option<f64> {
    let array = data.as_array();
    array.iter().fold(None, |acc, &x| {
        Some(acc.map_or(x, |a| a.max(x)))
    })
}

#[pyfunction]
fn array_min(data: PyReadonlyArray1<f64>) -> Option<f64> {
    let array = data.as_array();
    array.iter().fold(None, |acc, &x| {
        Some(acc.map_or(x, |a| a.min(x)))
    })
}

#[pyfunction]
fn array_std(data: PyReadonlyArray1<f64>) -> f64 {
    data.as_array().std(0.0)
}

#[pyfunction]
fn array_var(data: PyReadonlyArray1<f64>) -> f64 {
    data.as_array().var(0.0)
}

#[pyfunction]
fn linspace<'py>(py: Python<'py>, start: f64, stop: f64, num: usize) -> &'py PyArray1<f64> {
    if num == 0 {
        return Array1::from(vec![]).into_pyarray(py);
    }
    
    if num == 1 {
        return Array1::from(vec![start]).into_pyarray(py);
    }
    
    let step = (stop - start) / (num - 1) as f64;
    let result: Vec<f64> = (0..num)
        .map(|i| start + i as f64 * step)
        .collect();
    
    Array1::from(result).into_pyarray(py)
}

#[pyfunction]
fn zeros<'py>(py: Python<'py>, shape: (usize, usize)) -> &'py PyArray2<f64> {
    Array2::zeros(shape).into_pyarray(py)
}

#[pyfunction]
fn ones<'py>(py: Python<'py>, shape: (usize, usize)) -> &'py PyArray2<f64> {
    Array2::ones(shape).into_pyarray(py)
}

#[pyfunction]
fn eye<'py>(py: Python<'py>, n: usize) -> &'py PyArray2<f64> {
    Array2::eye(n).into_pyarray(py)
}

// 모듈에 함수들 등록
pub fn register_numpy_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_to_ndarray, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(standardize, m)?)?;
    m.add_function(wrap_pyfunction!(fillna_with_mean, m)?)?;
    m.add_function(wrap_pyfunction!(filter_positive, m)?)?;
    m.add_function(wrap_pyfunction!(sort_by_column, m)?)?;
    m.add_function(wrap_pyfunction!(hstack_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(vstack_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(to_float32, m)?)?;
    m.add_function(wrap_pyfunction!(expand_dims, m)?)?;
    
    // 벡터 연산
    m.add_function(wrap_pyfunction!(vector_add, m)?)?;
    m.add_function(wrap_pyfunction!(vector_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(vector_sum, m)?)?;
    m.add_function(wrap_pyfunction!(vector_mean, m)?)?;
    
    // 행렬 연산
    m.add_function(wrap_pyfunction!(matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    
    // 통계 함수
    m.add_function(wrap_pyfunction!(array_max, m)?)?;
    m.add_function(wrap_pyfunction!(array_min, m)?)?;
    m.add_function(wrap_pyfunction!(array_std, m)?)?;
    m.add_function(wrap_pyfunction!(array_var, m)?)?;
    
    // 배열 생성 함수
    m.add_function(wrap_pyfunction!(linspace, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let array = Array1::from(data).into_pyarray(py);
            let result = normalize(py, array.readonly());
            let result_vec: Vec<f64> = result.to_vec().unwrap();
            
            let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
            for (a, b) in result_vec.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        });
    }

    #[test]
    fn test_vector_operations() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let a = Array1::from(vec![1.0, 2.0, 3.0]).into_pyarray(py);
            let b = Array1::from(vec![4.0, 5.0, 6.0]).into_pyarray(py);
            
            let result = vector_add(py, a.readonly(), b.readonly()).unwrap();
            let result_vec: Vec<f64> = result.to_vec().unwrap();
            assert_eq!(result_vec, vec![5.0, 7.0, 9.0]);
        });
    }
}