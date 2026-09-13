# Chunked Execution Run Plan for Vision Pipeline Evaluation

> **Notice:** Time estimates below are **refined from measured quick-mode data** (293.1 s/task measured on Qwen2.5-VL-3B-Instruct).

## Overview

This execution plan divides the 16 representative samples across all 10 manifest categories into 5 independent 10–15 minute sessions using `--category`, `--quick`, and `--resume`.

| Session | Focus Area | Categories | Samples | Tasks | Est. Time |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | Core Process Drawings | `PID,PFD` | 4 | 4 | ~20 min |
| 2 | Equipment & Instrumentation | `Equipment_Drawings,Instrumentation` | 3 | 3 | ~15 min |
| 3 | Rotating & Electrical | `Pump_Diagrams,Electrical` | 2 | 2 | ~10 min |
| 4 | Defect Inspection | `Corrosion_Defects,Defect_Detection` | 4 | 4 | ~20 min |
| 5 | Photovoltaic & Notes | `Infrared,Handwritten_Notes` | 3 | 3 | ~15 min |
| **Total** | **All 10 Categories** | — | **16** | **16** | **~78 min** |

---

## Session Commands

### Session 1: Core Process Drawings
```bash
python evaluate_vision.py --category PID,PFD --quick --resume --device auto --max-run-minutes 15 --threads 4
```

### Session 2: Equipment & Instrumentation
```bash
python evaluate_vision.py --category Equipment_Drawings,Instrumentation --quick --resume --device auto --max-run-minutes 15 --threads 4
```

### Session 3: Rotating & Electrical
```bash
python evaluate_vision.py --category Pump_Diagrams,Electrical --quick --resume --device auto --max-run-minutes 15 --threads 4
```

### Session 4: Defect Inspection
```bash
python evaluate_vision.py --category Corrosion_Defects,Defect_Detection --quick --resume --device auto --max-run-minutes 15 --threads 4
```

### Session 5: Photovoltaic & Notes
```bash
python evaluate_vision.py --category Infrared,Handwritten_Notes --quick --resume --device auto --max-run-minutes 15 --threads 4
```

---

## Step 6: Consolidation & Full Report

After completing all sessions, run the consolidation command without category filter:
```bash
python evaluate_vision.py --resume
```

This loads all accumulated inferences and generates final unified `vision_evaluation.json`, `vision_evaluation.csv`, and `README.md` artifacts.
