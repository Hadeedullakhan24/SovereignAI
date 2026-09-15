# datasets/

Flat structure — just drop files into the matching folder.

- pidqa/              -> PIDQA repo (images + qa_pairs, clone directly here)
- doclaynet/           -> DocLayNet subset (1000 pages + annotations)
- maintenance/         -> AI4I 2020 CSV, CMAPSS if used later
- safety_docs/         -> OISD + PNGRB PDFs, plus synthetic SOP sample
- inspection_reports/  -> synthetic + any real inspection reports
- templates/           -> approval note, work order, incident report templates
- emails/              -> synthetic email samples
- manuals/             -> OEM manuals (pump/valve/transmitter etc.)

Naming convention: prefix source, e.g. oisd_fire_safety.pdf, pngrb_pipeline_integrity.pdf
