# GSPARC CIKM 2026 Experiment Plan

## Overview

TPDS reviewer feedback + CIKM 2026 repositioning을 반영한 실험 계획.
논문의 메시지를 "GPU performance" → "scalable sparse data processing"로 전환.

실험 구성 순서: **Representation Effect → Execution Model Effect → Robustness/Scalability → End-to-end Runtime**

---

## Current Status (Completed Experiments)

| Experiment | Status | Result File |
|---|---|---|
| Accumulator Ablation (Dense/ESC/Adaptive) | Done | `result/Fig_ablation_accumulator.out` |
| Stage-wise Breakdown | Done | `result/Fig_stagewise_breakdown.out` |
| Summary & Charts | Done | `result/ablation_summary.txt`, `result/*.pdf` |

Key findings:
- Heuristic accuracy: **81.3%** (13/16), 3 MISS cases (Flickr-3m, Benzene, Guanine)
- SLITOM conversion dominates (96-98%) for large sparse tensors
- Contraction dominates (51-71%) for dense-output cases

---

## WS1: SLITOM vs COO Memory Footprint

**Purpose**: SLITOM format의 메모리 효율성 정량적 증명
**Addresses**: Reviewer 2 concern, CIKM representation effect

### Code Changes

**`include/gsparc/sptc.cuh`** (3 locations):
- `SpTC()` ~line 35 이후 (X_prtn_num 설정 직후)
- `multi_SpTC()` ~line 380 이후
- `extra::SpTC()` ~line 723 이후

각 위치에서 SLITOM 변환 완료 후 아래 출력 추가:
```cpp
// Memory footprint comparison
uint64_t slitom_bytes = SX->nnz * (sizeof(lindex_t) + sizeof(value_t))
                      + SY->nnz * (sizeof(lindex_t) + sizeof(value_t));
uint64_t coo_bytes = SX->nnz * (SX->nmodes * sizeof(uint64_t) + sizeof(value_t))
                   + SY->nnz * (SY->nmodes * sizeof(uint64_t) + sizeof(value_t));
fprintf(stderr, "--------Memory Footprint--------\n");
fprintf(stderr, "SLITOM bytes: %lu\n", slitom_bytes);
fprintf(stderr, "COO bytes: %lu\n", coo_bytes);
fprintf(stderr, "Compression ratio: %.2fx\n", (double)coo_bytes / slitom_bytes);
```

> Note: extra::SpTC의 경우 `ulindex_t` 추가 고려 필요:
> `slitom_bytes += (SX->nnz + SY->nnz) * sizeof(ulindex_t);`

### Script
- `scripts_local/Fig_memory_footprint.sh`: 전체 데이터셋 `-n 1`로 실행
- `scripts_local/parse_memory.py`: stderr에서 memory 정보 추출

### Expected Output
- Bar chart: SLITOM vs COO bytes per dataset
- Compression ratio table (예상: 3D=1.5x, 4D=2.0x, 5D=2.5x)

---

## WS2: Partition Ablation

**Purpose**: Mode-decoupled partitioning의 효과 증명
**Addresses**: Reviewer 4 concern, CIKM execution model

GSPARC에는 **2단계 파티셔닝**이 존재:
1. **Static Partition** (`tensor_manager.tpp:633`): SLITOM 변환 후 free mode 기준으로 X, Y 분할 → disjoint output 보장
2. **Dynamic Partition** (`contraction.cuh:110`): ESC contraction 시 GPU 메모리 부족하면 contract mode 기준 추가 분할

### Experiment A: Static Partition 수에 따른 성능 변화

**구현**:
- `cmdline_opts.hpp`: `int _forced_prtn_num;` 멤버 추가 + getter
- `cmdline_opts.cpp`: `-p` 플래그 파싱 (값: 0=auto, 1,2,4,8,16=강제 설정)
- `tensor_manager.tpp`: `FindPartitionNum()` 에서 `_forced_prtn_num > 0`이면 해당 값 사용

**실험 매트릭스**:
```
prtn_num = {1 (no partition), 2, 4, auto(default)}
datasets = {nell-2, uber, flickr, delicious, vast, T1(2137), benzene}
```

**측정 항목**:
- Total runtime
- Partition time
- Contraction time
- 파티션당 NNZ 분포 (load balance)
- 메모리 peak usage

**기대 결과**:
- prtn=1: 작은 텐서에서는 빠르지만, 큰 텐서에서 OOM 또는 성능 저하
- prtn=auto: 최적 또는 근최적 성능
- 파티션이 많아질수록 오버헤드 증가하지만 메모리 사용 감소

### Experiment C: Dynamic Partition 통계

**구현**:
- `contraction.cuh`의 `ESC_Contraction()` 에서 이미 `dprtn_timer` 존재
- Dynamic partition 발생 시 추가 통계 출력:
```cpp
fprintf(stderr, "--------Dynamic Partition--------\n");
fprintf(stderr, "Dynamic partitions: %d\n", dynamic_prtn_num);
fprintf(stderr, "Max partition ir_nnz: %lu\n", max_prtn_ir_nnz);
fprintf(stderr, "Dynamic partition time: %.4f ms\n", dprtn_timer->getTotalTime());
```

**측정 항목**:
- Dynamic partition 발생 횟수
- Dynamic partition 시간
- max_prtn_ir_nnz vs total ir_nnz 비율

**기대 결과**: 대부분 dynamic partition 없이 처리, 대규모 ESC에서만 발생 → 오버헤드 minimal

### Script
- `scripts_local/Fig_ablation_partition.sh`: 각 데이터셋 × {1, 2, 4, auto} 파티션 수 조합

---

## WS3: ESC Sort Overhead Breakdown

**Purpose**: ESC contraction 내부 sub-stage 시간 분석
**Addresses**: Reviewer 1 #4 (stage-wise), CIKM execution model detail

### Code Changes

**`include/gsparc/contraction.cuh`** `ESC_Contraction()` (~line 390-449):

```cpp
// 3 sub-stages with cudaEvent timers:
// 1. Kernel (lines 395-401): element-wise multiply + index compute
cudaEvent_t esc_k_start, esc_k_end;
cudaEventCreate(&esc_k_start); cudaEventCreate(&esc_k_end);
cudaEventRecord(esc_k_start);
// ... kernel launch ...
cudaEventRecord(esc_k_end);

// 2. RadixSort (lines 437-439)
cudaEvent_t esc_s_start, esc_s_end;
// ... CUB DeviceRadixSort ...

// 3. ReduceByKey (lines 441-444)
cudaEvent_t esc_r_start, esc_r_end;
// ... CUB DeviceReduceByKey ...

// Print sub-timers
float kernel_ms, sort_ms, reduce_ms;
cudaEventElapsedTime(&kernel_ms, esc_k_start, esc_k_end);
cudaEventElapsedTime(&sort_ms, esc_s_start, esc_s_end);
cudaEventElapsedTime(&reduce_ms, esc_r_start, esc_r_end);
fprintf(stderr, "ESC kernel: %.4f ms, Sort: %.4f ms, Reduce: %.4f ms\n",
        kernel_ms, sort_ms, reduce_ms);
```

동일한 변경:
- `ESC_Contraction_multi()` (~lines 557-660)
- `contraction_extra.cuh`의 `extra::ESC_Contraction()`

### Script
- `scripts_local/Fig_esc_breakdown.sh`: 전체 데이터셋에 `-d -1` (force ESC) 실행

### Expected Output
- Stacked bar chart: kernel / sort / reduce 비율
- Sort가 bottleneck이 아님을 증명 (sorting overhead 우려 해소)

---

## WS4: Large-Scale Tensor Experiments

**Purpose**: Billion-scale 텐서에서의 확장성 증명
**Addresses**: CIKM robustness/scalability

### Script
- `scripts_local/Fig_large_scale.sh`:

```bash
# Amazon (1.7B NNZ) - 2-mode
./build/gsparc -X amazon-reviews.tns -Y amazon-reviews.tns -c 2 -x 0 2 -y 0 2 -g {1,2,4,6} -n 1

# Patents - 2-mode
./build/gsparc -X patents.tns -Y patents.tns -c 2 -x 1 2 -y 1 2 -g {1,2,4,6} -n 1

# Reddit (4.7B NNZ) - 2-mode
./build/gsparc -X reddit-2015.tns -Y reddit-2015.tns -c 2 -x 0 2 -y 0 2 -g {1,2,4,6} -n 1
```

### Expected Output
- Multi-GPU scalability chart: runtime vs GPU count
- Memory footprint chart (WS1 출력 활용)
- Amazon SLITOM conversion 시간 별도 분석

---

## WS5: Heuristic Improvement (Optional)

**Purpose**: Ablation에서 발견된 3 MISS cases 개선
**Priority**: Low (현재 81.3% accuracy는 논문 제출 가능)

### Current Heuristic (`sptc.cuh:201-203`)
```cpp
bool default_dense = (dense_nnz < ir_nnz
    && memory_pool->is_available(dense_acc_size)
    && dense_nnz > sub_X_nnz);
```

### Proposed Improvement
```cpp
double output_density = (double)ir_nnz / dense_nnz;
bool default_dense = (output_density > DENSITY_THRESHOLD  // e.g., 0.5
    && memory_pool->is_available(dense_acc_size)
    && dense_nnz > sub_X_nnz);
```

### MISS Case Analysis
| Case | dense_nnz | ir_nnz | Current Decision | Optimal | Root Cause |
|---|---|---|---|---|---|
| Flickr 3-mode | - | - | Dense | ESC | Marginal (3.6% diff) |
| Benzene | ~1.9M | - | ESC | Dense | Structured sparsity, high output density |
| Guanine | ~24M | - | ESC | Dense | Same as Benzene |

---

## Implementation Order

| Priority | WS | Effort | Impact |
|---|---|---|---|
| 1 | WS1 (Memory Footprint) | Small (fprintf only) | High (CIKM representation) |
| 2 | WS3 (ESC Breakdown) | Medium (cudaEvent timers) | High (reviewer concern) |
| 3 | WS2 (Partition Ablation) | Medium (CLI flag + logic) | High (reviewer concern) |
| 4 | WS4 (Large-Scale) | None (script only) | Medium (scalability) |
| 5 | WS5 (Heuristic) | Small (condition change) | Low (optional) |

---

## Files to Modify

| File | WS | Changes |
|---|---|---|
| `include/gsparc/sptc.cuh` | WS1 | Memory footprint logging (3 locations) |
| `include/gsparc/contraction.cuh` | WS3 | ESC sub-timers (`ESC_Contraction`, `ESC_Contraction_multi`) |
| `include/gsparc/contraction_extra.cuh` | WS3 | ESC sub-timers (`extra::ESC_Contraction`) |
| `include/gsparc/cmdline_opts.hpp` | WS2 | `-p` flag declaration |
| `src/gsparc/cmdline_opts.cpp` | WS2 | `-p` flag parsing |
| `include/gsparc/tensor_manager.tpp` | WS2 | Forced partition num logic in `FindPartitionNum()` |

## New Files

| File | WS | Purpose |
|---|---|---|
| `scripts_local/Fig_memory_footprint.sh` | WS1 | Memory footprint experiment |
| `scripts_local/Fig_ablation_partition.sh` | WS2 | Partition ablation (prtn={1,2,4,auto} + dynamic stats) |
| `scripts_local/Fig_esc_breakdown.sh` | WS3 | ESC sub-stage breakdown |
| `scripts_local/Fig_large_scale.sh` | WS4 | Large-scale multi-GPU scalability |
| `scripts_local/parse_memory.py` | WS1 | Parse memory footprint output |

---

## Verification Checklist

1. [ ] `cd build && cmake .. && make -j$(nproc)` — 컴파일 에러 없음
2. [ ] Smoke test: nell-2 데이터셋으로 새 플래그/출력 확인
3. [ ] WS1: Memory footprint 출력 형식 확인
4. [ ] WS2: `-p 1` vs default 비교, 파티션 수 변화 확인
5. [ ] WS3: ESC sub-timer 출력 확인
6. [ ] WS4: Large-scale 실험 실행 (장시간 소요)
7. [ ] 전체 결과 파싱 및 차트 생성
