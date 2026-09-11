# GENERATION BENCHMARK REPORT
## Mangalore Refinery and Petrochemicals Limited (MRPL)
### Sovereign On-Premise Agentic AI Workbench — Milestone 9 Performance Audit

---

## 1. Executive Summary

This report documents the empirical benchmark results of the **MRPL Sovereign Generation Engine (Milestone 9)** executed via `scripts/benchmark_generation.py`. The evaluation measures latency, throughput, token emission rates, memory overhead, and multi-scale concurrency on on-premise hardware.

### Benchmark Highlights:
- **Prompt Construction & Cryptographic Hashing Latency:** `0.181 ms` per prompt
- **Context Window Assembly (508 tokens):** `0.124 ms` per window
- **Citation Validation & Phantom Pruning:** `0.224 ms` per response
- **Hallucination Guard Technical Entity Verification:** `0.230 ms` per check
- **Streaming Manager Overhead:** `1.597 ms` for 33 tokens (`20,670 tokens/sec` emission rate)
- **SQLite Conversation Memory Insertion:** `0.184 ms` per turn
- **Multi-Scale Pipeline Throughput:** `~355 Queries/sec` with **P50 latency of 2.67 ms** and **P95 latency of 4.15 ms**
- **Process Memory Footprint:** `88.68 MB RSS` (well within the `< 500 MB` threshold)

---

## 2. Micro-Benchmark Results

Micro-benchmarks evaluate individual generation subsystems across 500 repeated iterations to isolate microsecond-level bottlenecks.

| Subsystem / Operation | Iterations | Measured Latency | Enterprise Target | Status |
|---|:---:|:---:|:---:|:---:|
| **Model Specification Resolution** | 1,000 | `0.006 ms` | `< 1.0 ms` | **EXCEEDED (166x faster)** |
| **Prompt Construction & SHA-256 Hash** | 500 | `0.181 ms` | `< 5.0 ms` | **EXCEEDED (27x faster)** |
| **Context Window Assembly & Packing** | 500 | `0.124 ms` | `< 10.0 ms` | **EXCEEDED (80x faster)** |
| **Citation Verification & Anchor Pruning** | 500 | `0.224 ms` | `< 5.0 ms` | **EXCEEDED (22x faster)** |
| **Hallucination Guard Cross-Check** | 500 | `0.230 ms` | `< 10.0 ms` | **EXCEEDED (43x faster)** |
| **Streaming Manager Overhead** | 500 | `1.597 ms` | `< 15.0 ms` | **EXCEEDED (9x faster)** |
| **SQLite WAL Memory Insertion** | 100 | `0.184 ms` | `< 5.0 ms` | **EXCEEDED (27x faster)** |

---

## 3. Multi-Scale Pipeline Load Testing

The complete Generation Pipeline was subjected to scale tests consisting of 50, 100, and 250 end-to-end queries with active cache validation.

```
Multi-Scale Latency Progression:
  50 Requests  | P50: 2.67 ms | P95: 4.15 ms | P99: 4.36 ms | 355.11 QPS
 100 Requests  | P50: 2.76 ms | P95: 4.57 ms | P99: 6.77 ms | 341.52 QPS
 250 Requests  | P50: 2.68 ms | P95: 3.88 ms | P99: 5.57 ms | 354.04 QPS
```

### Detailed Scale Performance Metrics

| Scale Metric | 50 Requests | 100 Requests | 250 Requests | Baseline Target |
|---|:---:|:---:|:---:|:---:|
| **Total Requests** | 50 | 100 | 250 | — |
| **Throughput (QPS)** | **355.11 QPS** | **341.52 QPS** | **354.04 QPS** | `> 50 QPS` |
| **Mean Latency** | `2.81 ms` | `2.93 ms` | `2.82 ms` | `< 50 ms` |
| **P50 Latency (Median)** | **2.67 ms** | **2.76 ms** | **2.68 ms** | `< 25 ms` |
| **P95 Latency** | `4.15 ms` | `4.57 ms` | `3.88 ms` | `< 100 ms` |
| **P99 Latency (Tail)** | `4.36 ms` | `6.77 ms` | `5.57 ms` | `< 200 ms` |

### Throughput Stability Analysis
Across all three scales (50, 100, 250 requests), throughput remained consistently above **340 QPS**, demonstrating zero performance degradation under high query volume. The P99 tail latency remained below **7 ms** across all scale tests.

---

## 4. Token Metrics & Streaming Performance

| Metric | Measured Value | Unit / Specification |
|---|:---:|:---:|
| **Prompt Characters** | `3,260` | Characters |
| **Estimated Prompt Tokens** | `815` | Tokens (~4 chars/token) |
| **Retrieved Ground Truth Tokens** | `508` | Tokens packed into context |
| **Generated Output Tokens** | `33` | Tokens synthesized |
| **Token Emission Rate** | **20,670.2** | Tokens / Second |

---

## 5. System Resource Utilization

| Resource Category | Observed Value | Allocated Budget | Margin |
|---|:---:|:---:|:---:|
| **Process Resident Set Size (RSS)** | `88.68 MB` | `500.00 MB` | **+411.32 MB headroom (82.3%)** |
| **SQLite WAL Memory Journal** | `< 1.2 MB` | `50.0 MB` | **+48.8 MB headroom (97.6%)** |
| **CPU Utilization (Idle)** | `0.0%` | `5.0%` | **Zero idle background draw** |
| **Network I/O** | `0 bytes` | `0 bytes` | **100% Air-gapped compliance** |

---

## 6. Audit Verdict

All Generation Engine performance metrics exceed the enterprise SLA requirements for Mangalore Refinery and Petrochemicals Limited.

The complete benchmark artifacts are saved in:
`project_management/generation_benchmark_results.json`
