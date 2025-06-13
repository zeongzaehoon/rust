## 1. Rust 설치
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
- linux, mac 동일

```text
1) Proceed with installation (default)
2) Customize installation
3) Cancel installation
```
- 1 누르고 enter.

```bash
# Cargo 환경변수 적용
. "$HOME/.cargo/env"

# 설치되었는지 확인
rustc --version
cargo --version
```

## 2. 파이썬과 연동
- maturin: 파이썬에서 rust 연동하게 해주는 빌드 도구 라이브러리.
```bash
pip install maturin
```
- rust 프로젝트를 파이썬 최상위 스크립트와 동일한 곳에 생성
```bash
cargo new --lib rust
cd rust

cat cargo.toml
```
```txt
[package]
name = "rust"
version = "0.1.0"
edition = "2025"

[lib]
name = "rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
```

- 코드 작성 후 러스트 - 파이썬 동기 빌드
```bash
maturin develop
```

## 3. 파이썬, 러스트 역할
|언어|장점|사용예시|
|-------|-------|-------|
|Python|생산성, 생태계, LLM/ML 등과의 연동 탁월|서비스 로직, 모델 호출, API 처리
|Rust|속도, 안전성, 병렬 처리, 메모리 절약|고속 연산, 반복 로직, 데이터 파싱 등

- 내 생각: 아래 구조가 두 언어의 장점을 가장 잘쓰는 구조 같음 
	+ 파이썬: IO로 데이터를 받고, 반환해주는 통로.
	+ 러스트: 데이터를 파싱, 연산 -> 메모리 관리와 다중처리 가능하기 때문