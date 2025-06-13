"""
Rust로 구현된 NumPy 유사 함수들을 위한 Python 래퍼
"""
from rust import numpy as rust
import numpy as np
from typing import List, Union, Optional

ArrayLike = Union[List[float], List[int], np.ndarray]

class RustNumpy:
    """Rust 백엔드를 사용하는 NumPy 유사 클래스"""
    
    @staticmethod
    def list_to_vec(data: ArrayLike) -> List[float]:
        """리스트를 벡터로 변환 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.list_to_vec(data)
    
    @staticmethod
    def flatten_to_matrix(data: ArrayLike, rows: int, cols: int) -> List[List[float]]:
        """1D 배열을 2D 행렬로 변환 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.flatten().tolist()
        return rust.flatten_to_matrix(data, rows, cols)
    
    @staticmethod
    def normalize(data: ArrayLike) -> List[float]:
        """Min-Max 정규화 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.normalize(data)
    
    @staticmethod
    def standardize(data: ArrayLike) -> List[float]:
        """Z-score 표준화 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.standardize(data)
    
    @staticmethod
    def fillna_with_mean(data: List[Optional[float]]) -> List[float]:
        """NaN 값을 평균으로 채우기 (Rust 처리)"""
        return rust.fillna_with_mean(data)
    
    @staticmethod
    def filter_positive(data: ArrayLike) -> List[float]:
        """양수만 필터링 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.filter_positive(data)
    
    @staticmethod
    def sort_by_column(matrix: List[List[float]], col: int = 0) -> List[List[float]]:
        """특정 컬럼으로 정렬 (Rust 처리)"""
        return rust.sort_by_column(matrix, col)
    
    @staticmethod
    def hstack_arrays(*arrays: List[List[float]]) -> List[List[float]]:
        """배열들을 수평으로 연결 (Rust 처리)"""
        return rust.hstack_arrays(list(arrays))
    
    @staticmethod
    def vstack_arrays(*arrays: List[List[float]]) -> List[List[float]]:
        """배열들을 수직으로 연결 (Rust 처리)"""
        return rust.vstack_arrays(list(arrays))
    
    @staticmethod
    def to_float32(data: ArrayLike) -> List[float]:
        """float32로 변환 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.to_float32(data)
    
    @staticmethod
    def expand_dims(data: ArrayLike, axis: int = 0) -> List[List[float]]:
        """차원 확장 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.expand_dims(data, axis)
    
    # 추가 벡터 연산 함수들
    @staticmethod
    def vector_add(a: ArrayLike, b: ArrayLike) -> List[float]:
        """벡터 덧셈 (Rust 처리)"""
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        return rust.vector_add(a, b)
    
    @staticmethod
    def vector_multiply(a: ArrayLike, b: ArrayLike) -> List[float]:
        """벡터 곱셈 (Rust 처리)"""
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        return rust.vector_multiply(a, b)
    
    @staticmethod
    def scalar_multiply(data: ArrayLike, scalar: float) -> List[float]:
        """스칼라 곱셈 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.scalar_multiply(data, scalar)
    
    @staticmethod
    def vector_sum(data: ArrayLike) -> float:
        """벡터 합계 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.vector_sum(data)
    
    @staticmethod
    def vector_mean(data: ArrayLike) -> Optional[float]:
        """벡터 평균 (Rust 처리)"""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        return rust.vector_mean(data)

# 편의를 위한 글로벌 함수들
def normalize_rust(data: ArrayLike) -> List[float]:
    return RustNumpy.normalize(data)

def standardize_rust(data: ArrayLike) -> List[float]:
    return RustNumpy.standardize(data)

def filter_positive_rust(data: ArrayLike) -> List[float]:
    return RustNumpy.filter_positive(data)

# 성능 비교를 위한 함수
def compare_performance():
    """NumPy vs Rust 성능 비교"""
    import time
    import numpy as np
    
    # 테스트 데이터
    data = np.random.randn(100000).tolist()
    
    # NumPy 버전
    start = time.time()
    np_result = (np.array(data) - np.array(data).min()) / (np.array(data).max() - np.array(data).min())
    numpy_time = time.time() - start
    
    # Rust 버전
    start = time.time()
    rust_result = RustNumpy.normalize(data)
    rust_time = time.time() - start
    
    print(f"NumPy 시간: {numpy_time:.6f}초")
    print(f"Rust 시간: {rust_time:.6f}초")
    print(f"성능 향상: {numpy_time/rust_time:.2f}배")
    
    return numpy_time, rust_time

if __name__ == "__main__":
    # 테스트 실행
    print("=== Rust NumPy 테스트 ===")
    
    # 기본 테스트
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"원본 데이터: {data}")
    print(f"정규화: {RustNumpy.normalize(data)}")
    print(f"표준화: {RustNumpy.standardize(data)}")
    print(f"양수 필터: {RustNumpy.filter_positive([-1, 0, 1, 2, 3])}")
    
    # 성능 비교
    print("\n=== 성능 비교 ===")
    compare_performance()