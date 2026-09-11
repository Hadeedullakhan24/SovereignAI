# SYNTHETIC INDUSTRIAL INSPECTION REPORT — CENTRIFUGAL PUMP

> Dataset status: Synthetic training/evaluation record for the Sovereign On-Premise Agentic AI Workbench.
> Not an MRPL record. Not for operational or safety decisions.

## 1. Document Control
| Field | Value |
|---|---|
| Report ID | INSP-P-005 |
| Equipment ID | P-315B |
| Equipment | Horizontal centrifugal process pump |
| Service | Hydrocarbon transfer |
| Inspection Type | Condition-based inspection |
| Inspection Date | 2026-08-20 |
| Report Status | FAIL — MAINTENANCE REQUIRED |

## 2. Operating Snapshot
| Parameter | Reading |
|---|---:|
| Suction pressure | 2.1 barg |
| Discharge pressure | 8.4 barg |
| Flow | 118 m³/h |
| DE bearing vibration | 8.7 mm/s RMS |
| NDE bearing vibration | 6.1 mm/s RMS |
| DE bearing temperature | 84 °C |
| NDE bearing temperature | 71 °C |
| Motor current | 91 A |
| Seal leakage | Visible intermittent wetting |

Acceptance limits must be retrieved from the applicable controlled pump/vendor documentation; values here are not universal limits.

## 3. Findings
### Vibration Trend
| Date | DE Vibration |
|---|---:|
| 2026-05-10 | 3.8 mm/s RMS |
| 2026-06-14 | 4.4 mm/s RMS |
| 2026-07-19 | 5.9 mm/s RMS |
| 2026-08-20 | 8.7 mm/s RMS |

The trend is increasing.

### Hydraulic Performance
Discharge pressure and flow are below the recent baseline. Potential causes requiring verification include hydraulic restriction, impeller condition, suction-side condition, internal wear, process change, or instrumentation error.

### Mechanical Seal
Intermittent external wetting was observed around the mechanical-seal area. No uncontrolled spray or major release was observed.

### Bearing Condition
DE bearing temperature is elevated and correlates with the vibration increase. Lubrication and bearing mechanical condition should be verified.

## 4. Risk Assessment
**FAIL — MAINTENANCE REQUIRED**

The combination of increasing vibration, elevated bearing temperature, reduced hydraulic performance and abnormal seal wetting requires controlled maintenance intervention.

## 5. Recommended Actions
1. Notify responsible operations and maintenance personnel.
2. Evaluate whether continued operation is permitted by the approved procedure.
3. Verify readings using calibrated instrumentation where appropriate.
4. Check standby pump availability.
5. Inspect DE/NDE bearings.
6. Verify lubrication against the vendor manual.
7. Check shaft alignment and coupling condition.
8. Inspect mechanical seal.
9. Inspect impeller/wear components if opened.
10. Check suction strainer/flow path.
11. Perform post-maintenance vibration measurement.

## 6. Agentic Reasoning Requirements
Before drafting a work order, retrieve the local pump manual, applicable maintenance/SOP documents, recent maintenance history, and relevant safety/work-permit requirements. Do not invent vibration limits or replacement intervals.

## 7. Approval
| Role | Status |
|---|---|
| Inspection | Completed |
| Maintenance | Action required |
| Engineering | Review required |
| Operations | Approval required |
| Work-order release | Human approval required |
