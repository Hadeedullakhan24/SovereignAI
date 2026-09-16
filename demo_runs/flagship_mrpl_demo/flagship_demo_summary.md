# SovereignAI — MRPL SIH 26117 Flagship Demo Execution Report

**Execution Date:** 2026-09-16 00:00:31 UTC  
**Scenario:** *Analyse the inspection report, refer to the local safety SOP, calculate the maintenance cost from the Excel sheet, and prepare an approval note.*  
**Target Asset:** Pressure Vessel V-2201 (Knockout Drum)  
**Deployment:** 100% Air-Gapped On-Premise Industrial Workbench  
**Total Pipeline Execution Time:** 212.63 seconds  

---

## 1. 17-Step Execution Scorecard

| Step | Operation | Status | Output / Artifact | Latency (ms) |
| :---: | :--- | :---: | :--- | :---: |
| **01** | Input Staging & Ingestion | 🟢 PASS | Report, Photo, Matrix staged | 31.8 |
| **02** | Document Type & MIME Detection | 🟢 PASS | Markdown, PNG, OpenXML XLSX | 0.0 |
| **03** | Multimodal OCR Extraction | 🟢 PASS | 1,612 characters parsed | 4146.6 |
| **04** | Local VLM Visual Inspection | 🟢 PASS | Surface & nozzle condition verified | 81971.3 |
| **05** | Structured Findings Extraction | 🟢 PASS | P=10.5 kg/cm²g, min shell=11.2 mm | 0.0 |
| **06** | Multi-Model Routing Dispatch | 🟢 PASS | RAG, Calculator, Docx routed | 31.6 |
| **07** | Autonomous Multi-Step Planning | 🟢 PASS | 7-step plan graph generated | 2.6 |
| **08** | Hybrid RAG Local SOP Search | 🟢 PASS | OISD-130 / API 510 citations retrieved | 125716.3 |
| **09** | Excel Maintenance Cost Parsing | 🟢 PASS | 6 line items, subtotal INR 254,300.00 | 54.5 |
| **10** | Deterministic Calculation | 🟢 PASS | t_min=5.3998 mm, budget=INR 315,077.70 | 1.9 |
| **11** | Math Reduction Trace Verification | 🟢 PASS | 6 verified reduction steps | 0.0 |
| **12** | Technical Synthesis & Citations | 🟢 PASS | Executive approval synthesis | 0.0 |
| **13** | Word (.docx) Deliverable Generation | 🟢 PASS | `V-2201_Statutory_Approval_Note.docx` (39,023 bytes) | 622.4 |
| **14** | Artifact Integrity Validation | 🟢 PASS | PK zip, document.xml, 0 placeholders | 41.5 |
| **15** | Human-in-the-Loop Approval Gate | 🟢 PASS | Checkpoint signed by Chief Inspector | 4.4 |
| **16** | Cryptographic Audit Trail | 🟢 PASS | Appended to `logs/agent_audit.jsonl` | 4.7 |
| **17** | Sovereignty & Airgap Proof | 🟢 PASS | 0 external calls (100% compliant) | 0.5 |

---

## 2. Engineering Verification Results

- **ASME Sec VIII Div 1 Minimum Required Thickness:** `5.3998 mm`
- **Actual Measured Shell Wall Thickness:** `11.2 mm`
- **Excess Statutory Safety Margin:** `+5.8002 mm`
- **Remaining Corrosion Allowance:** `78%`
- **Turnaround Authorized Budget:** `INR 315,077.70` (Base: INR 254,300.00 + 5% Contingency + 18% GST)
- **Authorized Inspector Sign-Off:** `Er. S. Menon (Chief Mechanical Inspector #AI-9021)` (Approved for service through Sep-2031)

---

## 3. Generated Deliverables in `demo_runs/flagship_mrpl_demo/`

1. [`V-2201_Statutory_Approval_Note.docx`](./V-2201_Statutory_Approval_Note.docx) — Formal Word report deliverable
2. [`maintenance_cost_matrix.xlsx`](./sandbox/maintenance_cost_matrix.xlsx) — Excel maintenance cost ledger
3. [`offline_compliance_report.json`](./offline_compliance_report.json) — Zero-external-network compliance audit
4. [`MRPL_CHK_V2201_STATUTORY_GATE_completed.json`](./MRPL_CHK_V2201_STATUTORY_GATE_completed.json) — Signed human approval gate checkpoint
5. [`agent_audit.jsonl`](./agent_audit.jsonl) — Cryptographic JSONL audit trail
