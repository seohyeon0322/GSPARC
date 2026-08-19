# GSPARC Sort Optimization Summary

## Overview

SLITOM 정렬(sort) 단계의 성능을 최적화하기 위한 변경 사항 정리.
Stage-wise breakdown 분석에서 Sort가 SLITOM 생성 시간의 주요 비중을 차지하는 것으로 확인.

---

## 구현된 최적화

### OPT-B: 128-bit 2-pass CUB RadixSort (핵심 최적화)

**파일:** `include/gsparc/sort.cuh` (sort_128, radix_sort_128_single), `include/gsparc/tensor_manager.tpp` (convert_and_sort_128)

**변경 내용:**
- 기존: Thrust `indirect sort` (comparison-based, O(n log n))
- 변경: 2-pass CUB RadixSort (O(n * bits / radix_width))

**알고리즘:**
1. Identity permutation 초기화: `[0, 1, 2, ..., nnz-1]`
2. **Pass 1:** CUB RadixSort by lower 64-bit (`d_indices`), permutation을 satellite value로 운반
3. **Pass 2:** (upper_bits > 0일 때) Gather `uindices` by permutation, CUB RadixSort by upper bits
4. **Final gather:** 최종 permutation으로 3개 배열 (uindices, indices, values) 재배치
5. cudaMemcpy(DeviceToDevice)로 원본 배열에 복사

**개선 효과 (128-bit datasets):**

| 텐서 | 기존 SLITOM | 최적화 SLITOM | Speedup |
|------|-----------|-------------|---------|
| flickr-4d (2-mode) | 1.738s | 1.136s | **1.53x** |
| flickr-4d (3-mode) | 2.626s | 0.978s | **2.68x** |
| delicious-4d (2-mode) | 2.449s | 1.479s | **1.66x** |
| delicious-4d (3-mode) | 3.575s | 1.316s | **2.72x** |

### OPT-A: 64-bit RadixSort 비트폭 최적화 — 롤백

**결과:** CUDA 13.0 + sm_120 환경에서 CUB DeviceRadixSort의 in-place 사용 시 특정 `end_bit` 값에서 illegal memory access (code=700) 발생.
- `sizeof(lindex_t) * 8 = 64` → 정상 동작
- `SX->nbits` (예: 35) → 일부 텐서(uber.tns)에서 CUDA error 700

**조치:** 64-bit path에서는 기존과 동일하게 `end_bit = 64` 유지. CUB 버그로 추정.

### 분석 후 스킵한 최적화

#### OPT-C: X==Y 중복 정렬 제거 — 스킵
- **이유:** X와 Y의 contraction mode 순서가 다를 수 있어 SLITOM 변환 결과가 상이

#### OPT-D: 불필요한 D2H 왕복 제거 — 스킵
- **이유:** Sort 후 `Partition()`이 host 메모리의 `slitom->indices`를 읽으므로 D2H 필수

---

## 추가 변경 (인프라)

### Makefile 수정
- `CPPFLAGS`에 `$(CUDA_INC_DIR)` 추가: `.cpp` 파일의 CUDA 헤더 포함
- `-lopenblas -lgfortran` 제거: 미사용 의존성
- `arch=sm_75` → `arch=sm_120`: Blackwell GPU (RTX PRO 6000) 지원

### sort_64: X/Y 독립 sort_bits
- `sort_bits_X = SX->nbits`, `sort_bits_Y = SY->nbits` 분리 (이후 64로 롤백되었으나 구조 개선)

---

## 실험 결과

### 64-bit Path 결과 (Sort/SLITOM 비율)

| 텐서 | NNZ | nbits | Sort | SLITOM | Sort/SLITOM |
|------|-----|-------|------|--------|-------------|
| nell-2 2m | 76.9M | 43 | 0.067s | 0.354s | 19.0% |
| nips 2m | 3.1M | 43 | 0.004s | 0.019s | 20.6% |
| chicago 2m | 5.3M | 30 | 0.006s | 0.022s | 26.6% |
| uber 2m | 3.3M | 35 | 0.003s | 0.016s | 21.3% |
| vast 2m | 26.0M | 47 | 0.023s | 0.212s | 11.0% |
| Qt1 4m | 33.8M | 30 | 0.028s | 0.552s | 5.0% |
| Qt2 4m | 194.3M | 30 | 0.160s | 4.101s | 3.9% |

### 128-bit Path 결과

| 텐서 | NNZ | nbits | Sort | SLITOM | Sort/SLITOM |
|------|-----|-------|------|--------|-------------|
| flickr 2m | 112.9M | 75 | 0.124s | 1.136s | 10.9% |
| flickr 3m | 112.9M | 75 | 0.123s | 0.978s | 12.6% |
| delicious 2m | 140.1M | 78 | 0.159s | 1.479s | 10.7% |
| delicious 3m | 140.1M | 78 | 0.161s | 1.316s | 12.2% |

### 기존 대비 SLITOM Speedup (128-bit, OPT-B 효과)

| 텐서 | Baseline | Optimized | Speedup |
|------|----------|-----------|---------|
| flickr 2m | 1.739s | 1.136s | **1.53x** |
| flickr 3m | 2.626s | 0.978s | **2.68x** |
| delicious 2m | 2.449s | 1.479s | **1.66x** |
| delicious 3m | 3.575s | 1.316s | **2.72x** |

### 대규모 텐서 결과

| 텐서 | NNZ | SLITOM | Total |
|------|-----|--------|-------|
| amazon | 1.74B | 60.38s | 986.55s |
| patents | 3.60B | 69.22s | 93.30s |
| reddit | 4.69B | 144.79s | 161.87s |

### 정확성 검증

모든 텐서에서 Result NNZ가 기존 결과와 일치 확인됨.
uber, benzene 등 이전 실험에서 CUDA error가 발생했던 데이터셋도 OPT-A 롤백 후 정상 동작.

---

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `include/gsparc/sort.cuh` | OPT-B (radix_sort_128_single 신규 함수 + sort_128 전면 재작성), sort_64 X/Y 독립 sort_bits |
| `include/gsparc/tensor_manager.tpp` | OPT-B (convert_and_sort_128 Thrust→CUB 전환) |
| `Makefile` | CUDA include path, OpenBLAS 제거, sm_120 |

## 실험 스크립트

| 파일 | 용도 |
|------|------|
| `scripts_local/Fig_sort_optimization.sh` | 전체 데이터셋 Sort 최적화 실험 |
| `scripts_local/parse_sort_optimization.py` | 실험 결과 파싱 |
| `scripts_local/compare_sort.py` | 기존 대비 비교 |
