# Next Milestone Plan — Milestone 3: Document Loaders & Deep Parsers

**Project:** Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)  
**Component:** Member 1 — Knowledge Base / RAG Engine  

---

## Milestone 3 Objectives

Now that the dataset is systematically validated and cataloged in `outputs/manifest.json`, Milestone 3 will implement concrete document loaders and tabular/section parsers conforming to `rag_engine.interfaces.BaseLoader`.

### Scope of Work
1. **PDF Loader (`rag_engine/loaders/pdf_loader.py`):**
   - Implement `BaseLoader` using `pypdf`.
   - Extract page numbers, headings, text content, and metadata.
2. **Word Document Loader (`rag_engine/loaders/docx_loader.py`):**
   - Implement `BaseLoader` using `python-docx`.
   - Extract paragraphs, tables, and document properties.
3. **Spreadsheet Loader (`rag_engine/loaders/xlsx_loader.py`, `csv_loader.py`):**
   - Implement `BaseLoader` using `openpyxl` / `csv`.
   - Parse tabular inspection records and equipment registers into structured rows.
4. **Text & Markdown Loader (`rag_engine/loaders/text_loader.py`, `markdown_loader.py`):**
   - Implement `BaseLoader` for SOPs, READMEs, and technical markdown docs.
5. **Loader Factory (`rag_engine/loaders/factory.py`):**
   - Automatic loader dispatch based on file extension and `manifest.json` metadata.
6. **Unit & Integration Tests:**
   - Comprehensive test suite for all loaders on real MRPL sample files.
