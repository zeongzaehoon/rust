import numpy as np
from typing import List, Tuple, Optional, Union

ArrayLike = Union[List[float], List[int], np.ndarray]



def list_to_ndarray(data: Union[List[int], List[float]]) -> np.ndarray:
	"""리스트를 NumPy 배열로 변환"""
	return np.array(data)

def flatten_to_matrix(arr: ArrayLike, rows: int, cols: int) -> np.ndarray:
    """1차원 배열 → 2차원 배열 reshape"""
    return np.reshape(np.array(arr), (rows, cols))

def normalize(arr: ArrayLike) -> np.ndarray:
    """정규화 (0 ~ 1)"""
    arr_np = np.array(arr, dtype=np.float32)
    return (arr_np - arr_np.min()) / (arr_np.max() - arr_np.min() + 1e-8)

def standardize(arr: ArrayLike) -> np.ndarray:
    """
    Z-score 표준화
    통계에서 데이터가 평균으로부터 얼마나 떨어져 있는지를 나타내는 값.
    """
    arr_np = np.array(arr, dtype=np.float32)
    return (arr_np - arr_np.mean()) / (arr_np.std() + 1e-8)

def fillna_with_mean(arr: ArrayLike) -> np.ndarray:
    """NaN 값을 평균으로 대체"""
    arr_np = np.array(arr, dtype=np.float32)
    mean_val = np.nanmean(arr_np)
    return np.where(np.isnan(arr_np), mean_val, arr_np)

def filter_positive(arr: ArrayLike) -> np.ndarray:
    """양수만 필터링"""
    arr_np = np.array(arr)
    return arr_np[arr_np > 0]

def sort_by_column(arr: ArrayLike, col: int = 0) -> np.ndarray:
    """특정 열 기준 정렬 (2D 배열)"""
    arr_np = np.array(arr)
    if arr_np.ndim != 2:
        raise ValueError("입력 배열은 2차원이어야 합니다.")
    return arr_np[arr_np[:, col].argsort()]

def hstack_arrays(*arrays: np.ndarray) -> np.ndarray:
    """수평 스텍"""
    return np.hstack(arrays)

def vstack_arrays(*arrays: np.ndarray) -> np.ndarray:
    """수직 스텍"""
    return np.vstack(arrays)

def to_float32(arr: ArrayLike) -> np.ndarray:
    """float32 타입으로 변환"""
    return np.array(arr, dtype=np.float32)

def expand_dims(arr: ArrayLike, axis: int = 0) -> np.ndarray:
    """차원 확장 (차원 추가)"""
    return np.expand_dims(np.array(arr), axis=axis)