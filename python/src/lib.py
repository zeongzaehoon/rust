def add(left: int, right: int) -> int:
    """두 정수를 더하는 함수"""
    return left + right


# 테스트 코드
if __name__ == "__main__":
    # 테스트 실행
    result = add(2, 2)
    assert result == 4, f"Expected 4, but got {result}"
    print("테스트 통과!")