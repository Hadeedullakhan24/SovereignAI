"""Comprehensive offline unit tests for Engineering Drawing Analyzer (drawing_analyzer.py).

Tests:
- ISA-5.1 instrument tag decomposition
- Equipment registry consolidation & OCR/VLM fusion
- Instrument registry extraction
- Proximity-based connectivity rejection vs. explicit evidence acceptance
- Cross-reference extraction (DWG, API, ASME, OISD)
- Title block metadata extraction
- RAG chunk export schema & hierarchy
- Agent queryable interface methods
- Fast runtime (<1 second execution)
"""
from __future__ import annotations


import time
from pathlib import Path

import pytest

from member3_ocr.core.drawing_analyzer import (
    AgentQueryableDrawing,
    ConnectionEntry,
    DrawingAnalysisRecord,
    DrawingAnalyzer,
    DrawingAnalyzerConfig,
    DrawingCrossReference,
    DrawingLabel,
    DrawingMetadata,
    DrawingRegion,
    DrawingType,
    EquipmentEntry,
    InstrumentEntry,
    SpatialRelation,
    SpatialRelationType,
    TitleBlockInfo,
    analyze_engineering_drawing,
    compute_aggregate_confidence,
    export_for_agent,
    export_for_rag,
    extract_connectivity,
    extract_drawing_cross_references,
    extract_drawing_title_block,
    extract_equipment_registry,
    extract_instrument_registry,
    is_trusted_confidence,
    parse_isa_instrument_tag,
)
from member3_ocr.core.ocr_pipeline import BoundingBox, TextBlock


def _make_bbox(left: float = 10.0, top: float = 10.0, right: float = 100.0, bottom: float = 50.0) -> BoundingBox:
    return BoundingBox(left=left, top=top, right=right, bottom=bottom, coordinate_space="processed_pixels")


def _make_label(label_id: str, text: str, label_type: str = "unknown", conf: float = 0.95) -> DrawingLabel:
    return DrawingLabel(
        label_id=label_id,
        text=text,
        bbox=_make_bbox(),
        confidence=conf,
        label_type=label_type,
        source="ocr",
    )


class TestISAInstrumentTagParsing:
    def test_standard_transmitters_and_indicators(self) -> None:
        desc, code, loop = parse_isa_instrument_tag("PT-101")
        assert code == "PT"
        assert loop == "101"
        assert "Pressure" in desc and "Transmitter" in desc

        desc, code, loop = parse_isa_instrument_tag("TI-204B")
        assert code == "TI"
        assert loop == "204B"
        assert "Temperature" in desc and "Indicator" in desc

    def test_specialized_valves_and_controllers(self) -> None:
        desc, code, loop = parse_isa_instrument_tag("FIC-202")
        assert desc == "Flow Indicating Controller"
        assert code == "FIC"
        assert loop == "202"

        desc, code, loop = parse_isa_instrument_tag("PSV-301")
        assert desc == "Pressure Safety Valve"
        assert code == "PSV"

        desc, code, loop = parse_isa_instrument_tag("MOV-105")
        assert desc == "Motor Operated Valve"
        assert code == "MOV"

    def test_fallback_unparsed_tag(self) -> None:
        desc, code, loop = parse_isa_instrument_tag("UNKNOWNTAG")
        assert desc == "Instrument"
        assert code == "UNKNOWNTAG"


class TestEquipmentRegistryConsolidation:
    def test_ocr_and_vlm_fusion_on_matching_tag(self) -> None:
        ocr_lbl = _make_label("l1", "P-205A", label_type="equipment_tag", conf=0.98)
        vlm_item = {
            "equipment_type": "Centrifugal Pump",
            "name_or_tag": "P-205A",
            "confidence": 0.85,
            "evidence": "Observed pump casing and discharge pipe",
        }
        entries = extract_equipment_registry([ocr_lbl], [vlm_item])
        assert len(entries) == 1
        e = entries[0]
        assert e.tag == "P-205A"
        assert e.equipment_type == "Centrifugal Pump"
        assert e.source == "both"
        assert e.confidence == 0.98

    def test_ocr_tag_without_vlm_tag_infers_type_from_prefix(self) -> None:
        ocr_lbl1 = _make_label("l1", "V-101", label_type="equipment_tag", conf=0.92)
        ocr_lbl2 = _make_label("l2", "E-102B", label_type="equipment_tag", conf=0.90)
        entries = extract_equipment_registry([ocr_lbl1, ocr_lbl2], [])
        assert len(entries) == 2
        e_map = {e.tag: e for e in entries}
        assert e_map["V-101"].equipment_type == "Vessel"
        assert e_map["V-101"].source == "ocr"
        assert e_map["E-102B"].equipment_type == "Heat Exchanger"

    def test_vlm_unbound_equipment_item_captured(self) -> None:
        vlm_item = {
            "equipment_type": "Distillation Column",
            "name_or_tag": None,
            "confidence": 0.70,
            "evidence": "Vertical tower visual outline",
        }
        entries = extract_equipment_registry([], [vlm_item])
        assert len(entries) == 1
        assert "DISTILLATION_COLUMN" in entries[0].tag
        assert entries[0].source == "vlm"


class TestInstrumentRegistryExtraction:
    def test_extract_instruments_from_ocr_and_vlm(self) -> None:
        ocr_lbl = _make_label("l1", "PT-101", conf=0.96)
        vlm_label = {"text": "FIC-202", "confidence": 0.80}
        instruments = extract_instrument_registry([ocr_lbl], [vlm_label])
        assert len(instruments) == 2
        i_map = {i.tag: i for i in instruments}
        assert i_map["PT-101"].instrument_type == "Pressure Transmitter"
        assert i_map["PT-101"].source == "ocr"
        assert i_map["FIC-202"].instrument_type == "Flow Indicating Controller"
        assert i_map["FIC-202"].source == "vlm"


class TestConnectivityExtractionAndProximityGuard:
    def test_rejects_proximity_based_connectivity_hallucinations(self) -> None:
        """Adversarial test: Proximity-based claims must be rejected and flagged."""
        prose = [
            "P-205 is connected to V-101 based on proximity.",
            "Valve V-12 connects to E-101 because they are adjacent.",
            "Assumed connection between pump P-101 and tank TK-01 due to proximity.",
        ]
        connections, flags = extract_connectivity(prose)
        assert len(connections) == 0, "Proximity claims must NOT produce connection edges"
        assert "unsupported_connected_to_relationship" in flags

    def test_rejects_geometric_connected_to_relations(self) -> None:
        """SpatialRelation with CONNECTED_TO and evidence='geometry' must be rejected."""
        reg1 = DrawingRegion("r1", "equipment", _make_bbox(10, 10, 50, 50), label="P-101")
        reg2 = DrawingRegion("r2", "equipment", _make_bbox(60, 10, 100, 50), label="V-101")
        geom_rel = SpatialRelation(
            relation_id="rel1",
            source_id="r1",
            relation=SpatialRelationType.CONNECTED_TO,
            target_id="r2",
            evidence="geometry",
        )
        connections, flags = extract_connectivity([], [geom_rel], [reg1, reg2])
        assert len(connections) == 0
        assert "unsupported_connected_to_relationship" in flags

    def test_accepts_evidenced_textual_and_line_connections(self) -> None:
        prose = [
            "Line L-042 connects P-205 to V-101",
            "Discharge from P-101 flows to E-102 via 4\"-P-101-CS",
            "LINE S-01: TK-101 -> P-201",
        ]
        connections, flags = extract_connectivity(prose)
        assert len(connections) == 3
        assert len(flags) == 0
        c_map = {(c.from_tag, c.to_tag): c for c in connections}
        assert ("P-205", "V-101") in c_map
        assert c_map[("P-205", "V-101")].evidence_type == "explicit_label"
        assert c_map[("P-205", "V-101")].line_tag == "L-042"


class TestDrawingCrossReferences:
    def test_extract_drawing_and_standard_references(self) -> None:
        text = [
            "REFER TO DWG-042 FOR INSTRUMENT DETAILS",
            "DESIGN AND FABRICATION PER API 650",
            "PIPING SPECIFICATION ACCORDING TO ASME B31.3",
            "SAFETY STANDARD OISD-STD-105 APPLIES",
            "SEE DRAWING NO. ENG-PID-9921",
        ]
        refs = extract_drawing_cross_references(text)
        targets = {r.target: r.ref_type for r in refs}
        assert "DWG-042" in targets and targets["DWG-042"] == "drawing"
        assert "ENG-PID-9921" in targets and targets["ENG-PID-9921"] == "drawing"
        assert any("API 650" in t for t in targets)
        assert any("ASME B31.3" in t for t in targets)
        assert any("OISD-STD-105" in t for t in targets)


class TestDrawingTitleBlockExtraction:
    def test_title_block_attributes_extraction(self) -> None:
        blocks = [
            "MRPL REFINERY PHASE III",
            "DRAWING NO: MRPL-PID-4091",
            "REV: 3B",
            "SCALE: NTS",
            "DATE: 12/04/2026",
            "UNIT: CDU-1",
        ]
        meta = DrawingMetadata(title="Crude Distillation Unit P&ID")
        tb = extract_drawing_title_block(blocks, meta)
        assert tb.drawing_number == "MRPL-PID-4091"
        assert tb.revision == "3B"
        assert tb.scale == "NTS"
        assert tb.date == "12/04/2026"
        assert tb.plant_unit == "CDU-1"
        assert tb.title == "Crude Distillation Unit P&ID"


class TestRAGAndAgentExports:
    @pytest.fixture
    def sample_record(self) -> DrawingAnalysisRecord:
        return DrawingAnalysisRecord(
            drawing_id="MRPL-PID-001",
            source_path="datasets/engineering_drawings/PID/sample.jpg",
            drawing_type=DrawingType.PID,
            title_block=TitleBlockInfo(
                drawing_number="MRPL-PID-001",
                revision="0",
                title="Refinery Crude Distillation",
                date="2026-09-13",
                scale="NTS",
                plant_unit="CDU-1",
            ),
            equipment=(
                EquipmentEntry(tag="P-205A", equipment_type="Centrifugal Pump", source="both", confidence=0.98),
                EquipmentEntry(tag="V-101", equipment_type="Flash Drum", source="both", confidence=0.95),
            ),
            instruments=(
                InstrumentEntry(tag="PT-101", instrument_type="Pressure Transmitter", function_code="PT", loop_number="101", source="ocr"),
                InstrumentEntry(tag="FIC-202", instrument_type="Flow Indicating Controller", function_code="FIC", loop_number="202", source="both"),
            ),
            connections=(
                ConnectionEntry(from_tag="P-205A", to_tag="V-101", line_tag="L-042", evidence_type="explicit_label", source_evidence="Line L-042 connects P-205A to V-101"),
            ),
            cross_references=(
                DrawingCrossReference(target="API 650", ref_type="standard", source_text="PER API 650"),
                DrawingCrossReference(target="DWG-042", ref_type="drawing", source_text="SEE DWG-042"),
            ),
        )

    def test_export_for_rag_creates_valid_chunks(self, sample_record: DrawingAnalysisRecord) -> None:
        chunks = export_for_rag(sample_record)
        assert len(chunks) == 4
        strategies = [c.metadata.chunk_strategy for c in chunks]
        assert "title_block" in strategies
        assert "equipment_list" in strategies
        assert "connectivity" in strategies
        assert "cross_references" in strategies

        # Validate breadcrumb and document id
        for chk in chunks:
            assert chk.metadata.document_id == "MRPL-PID-001"
            assert chk.metadata.heading_path[0] == "PID"
            assert chk.metadata.heading_path[1] == "MRPL-PID-001"
            assert len(chk.content) > 20

    def test_export_for_agent_query_methods(self, sample_record: DrawingAnalysisRecord) -> None:
        agent_view = export_for_agent(sample_record)
        assert agent_view.drawing_id == "MRPL-PID-001"
        assert agent_view.drawing_type == "PID"

        eq_list = agent_view.get_equipment_list()
        assert len(eq_list) == 2
        assert eq_list[0]["tag"] == "P-205A"

        conns = agent_view.get_connections_for("P-205A")
        assert len(conns) == 1
        assert conns[0]["to_tag"] == "V-101"

        cit = agent_view.citation_for_equipment("PT-101")
        assert cit is not None
        assert cit["tag"] == "PT-101"
        assert cit["instrument_type"] == "Pressure Transmitter"

        summary = agent_view.get_drawing_summary()
        assert "MRPL-PID-001" in summary
        assert "P-205A" in summary


class TestFastUnitExecutionTime:
    def test_synthetic_drawing_analysis_sub_second(self) -> None:
        """Assert complete mock drawing analysis pipeline executes in <100ms."""
        t0 = time.perf_counter()
        cfg = DrawingAnalyzerConfig()
        analyzer = DrawingAnalyzer(config=cfg)

        rec = analyzer.analyze_structured_drawing(
            "dummy_pid.png",
            drawing_type=DrawingType.PID,
            raw_vlm_text="Line L-01 connects P-101 to V-102. See DWG-100.",
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Analysis took {elapsed:.4f}s; must be <1.0s"
        assert rec.drawing_type == DrawingType.PID
        assert len(rec.connections) == 1
        assert len(rec.cross_references) == 1


class TestCategoryConsistencyAndLatencyScoping:
    def test_category_aliases_and_round_trip(self) -> None:
        from member3_ocr.evaluation.drawing.evaluate_drawing_analyzer import (
            CATEGORY_ALIASES,
            CATEGORY_DISPLAY_NAMES,
            MANIFEST_CATEGORIES,
            REPRESENTATIVE_DRAWINGS,
            filter_drawings_by_category,
        )

        assert frozenset(CATEGORY_ALIASES.keys()) == MANIFEST_CATEGORIES
        assert frozenset(CATEGORY_DISPLAY_NAMES.keys()) == MANIFEST_CATEGORIES

        # Test filtering by canonical name and aliases
        pid_filtered = filter_drawings_by_category(REPRESENTATIVE_DRAWINGS, ["PID"])
        assert len(pid_filtered) == 2
        assert all(d.category == "PID" for d in pid_filtered)

        pfd_alias_filtered = filter_drawings_by_category(REPRESENTATIVE_DRAWINGS, ["process flow"])
        assert len(pfd_alias_filtered) == 2
        assert all(d.category == "PFD" for d in pfd_alias_filtered)

        with pytest.raises(ValueError, match="Unknown category query"):
            filter_drawings_by_category(REPRESENTATIVE_DRAWINGS, ["invalid_category_xyz"])

    def test_latency_signature_scoping_prevents_blending(self) -> None:
        from member3_ocr.evaluation.drawing.evaluate_drawing_analyzer import (
            DrawingEvalRecord,
            LatencySignature,
        )

        sig_quick = LatencySignature("quick", 192, 24, True).to_key()
        sig_full = LatencySignature("full", 512, 128, True).to_key()
        assert sig_quick != sig_full

        records = [
            DrawingEvalRecord(
                sample_id="PID_01",
                category="PID",
                source_path="path/1",
                drawing_type="PID",
                drawing_number="DWG-01",
                title="T1",
                revision="0",
                scale="NTS",
                plant_unit="U1",
                equipment_count=2,
                instrument_count=1,
                connection_count=1,
                cross_reference_count=1,
                chunks_exported=4,
                hallucination_flags=[],
                connectivity_guard_passed=True,
                latency_ms=100.0,
                ocr_latency_ms=20.0,
                vlm_latency_ms=70.0,
                analyzer_latency_ms=10.0,
                settings_signature=sig_quick,
                timestamp="2026-09-13T00:00:00Z",
            ),
            DrawingEvalRecord(
                sample_id="PID_02",
                category="PID",
                source_path="path/2",
                drawing_type="PID",
                drawing_number="DWG-02",
                title="T2",
                revision="0",
                scale="NTS",
                plant_unit="U1",
                equipment_count=2,
                instrument_count=1,
                connection_count=1,
                cross_reference_count=1,
                chunks_exported=4,
                hallucination_flags=[],
                connectivity_guard_passed=True,
                latency_ms=1000.0,
                ocr_latency_ms=200.0,
                vlm_latency_ms=700.0,
                analyzer_latency_ms=100.0,
                settings_signature=sig_full,
                timestamp="2026-09-13T00:00:00Z",
            ),
        ]

        # Signature scoping: only average matching signature
        quick_records = [r for r in records if r.settings_signature == sig_quick]
        assert len(quick_records) == 1
        assert quick_records[0].latency_ms == 100.0
        avg_quick = sum(r.latency_ms for r in quick_records) / len(quick_records)
        assert avg_quick == 100.0, "Quick mode average must not blend with full mode latency"


class TestConfidenceCalibrationAndDefaults:
    """Test confidence defaults, None serialization, safe aggregation, and trust checks."""

    def test_unmeasured_confidence_defaults_to_none_and_serializes_to_null(self) -> None:
        """Unmeasured confidence fields must default to None and serialize to JSON null."""
        eq = EquipmentEntry(tag="P-101", equipment_type="Pump", source="ocr")
        inst = InstrumentEntry(tag="TI-101", instrument_type="Temp Ind", function_code="TI", loop_number="101")
        conn = ConnectionEntry(from_tag="P-101", to_tag="V-101")
        xref = DrawingCrossReference(target="DWG-001", ref_type="drawing")
        reg = DrawingRegion(region_id="r1", region_type="equipment", bbox=BoundingBox(0, 0, 10, 10))
        lbl = DrawingLabel(label_id="l1", text="P-101", bbox=BoundingBox(0, 0, 5, 5))
        rel = SpatialRelation(relation_id="rel1", source_id="r1", relation=SpatialRelationType.LEFT_OF, target_id="r2")

        assert eq.confidence is None
        assert inst.confidence is None
        assert conn.confidence is None
        assert xref.confidence is None
        assert reg.confidence is None
        assert lbl.confidence is None
        assert rel.confidence is None

        record = DrawingAnalysisRecord(
            drawing_id="TEST-001",
            source_path="test.png",
            drawing_type=DrawingType.PID,
            title_block=TitleBlockInfo(title="Test"),
            equipment=(eq,),
            instruments=(inst,),
            connections=(conn,),
            cross_references=(xref,),
        )

        d = record.to_dict()
        assert d["equipment"][0]["confidence"] is None
        assert d["instruments"][0]["confidence"] is None
        assert d["connections"][0]["confidence"] is None
        assert d["cross_references"][0]["confidence"] is None

        json_str = record.to_json()
        assert '"confidence": null' in json_str

    def test_aggregate_confidence_excludes_none_values(self) -> None:
        """Aggregate confidence must strictly exclude unmeasured (None) entries, not treat as 0 or crash."""
        eq1 = EquipmentEntry(tag="P-101", equipment_type="Pump", source="ocr", confidence=0.90)
        eq2 = EquipmentEntry(tag="P-102", equipment_type="Pump", source="ocr", confidence=None)
        eq3 = EquipmentEntry(tag="P-103", equipment_type="Pump", source="ocr", confidence=0.80)

        # Standard list aggregation
        agg = compute_aggregate_confidence([eq1, eq2, eq3])
        assert agg == 0.85, f"Expected 0.85 (mean of 0.90 and 0.80), got {agg}"

        # Record method
        rec = DrawingAnalysisRecord(
            drawing_id="TEST-002",
            source_path="test.png",
            drawing_type=DrawingType.PID,
            title_block=TitleBlockInfo(),
            equipment=(eq1, eq2, eq3),
        )
        assert rec.aggregate_confidence("equipment") == 0.85

        # All None entries should return None, not 0.0 or error
        all_unmeasured = [
            EquipmentEntry(tag="P-1", equipment_type="Pump", source="ocr", confidence=None),
            EquipmentEntry(tag="P-2", equipment_type="Pump", source="ocr", confidence=None),
        ]
        assert compute_aggregate_confidence(all_unmeasured) is None
        assert compute_aggregate_confidence([]) is None

    def test_trust_threshold_check_rejects_none_confidence(self) -> None:
        """A threshold trust check must treat None as 'unknown / do not trust' (False)."""
        assert is_trusted_confidence(None, threshold=0.5) is False
        assert is_trusted_confidence(None, threshold=0.0) is False
        assert is_trusted_confidence(0.40, threshold=0.5) is False
        assert is_trusted_confidence(0.50, threshold=0.5) is True
        assert is_trusted_confidence(0.95, threshold=0.5) is True

        # Test on AgentQueryableDrawing
        eq_high = EquipmentEntry(tag="P-101", equipment_type="Pump", source="both", confidence=0.95)
        eq_low = EquipmentEntry(tag="P-102", equipment_type="Pump", source="ocr", confidence=0.30)
        eq_none = EquipmentEntry(tag="P-103", equipment_type="Pump", source="ocr", confidence=None)

        rec = DrawingAnalysisRecord(
            drawing_id="TEST-003",
            source_path="test.png",
            drawing_type=DrawingType.PID,
            title_block=TitleBlockInfo(),
            equipment=(eq_high, eq_low, eq_none),
        )
        agent = export_for_agent(rec)
        trusted = agent.get_trusted_equipment(threshold=0.5)
        assert len(trusted) == 1
        assert trusted[0]["tag"] == "P-101"

        # Citation should note availability
        cit_none = agent.citation_for_equipment("P-103")
        assert cit_none is not None
        assert cit_none["confidence"] is None
        assert cit_none["confidence_available"] is False

        cit_high = agent.citation_for_equipment("P-101")
        assert cit_high is not None
        assert cit_high["confidence"] == 0.95
        assert cit_high["confidence_available"] is True

    def test_rag_export_formats_null_confidence_without_crashing(self) -> None:
        """export_for_rag must not throw TypeError formatting None confidence."""
        eq = EquipmentEntry(tag="P-101", equipment_type="Pump", source="ocr", confidence=None)
        inst = InstrumentEntry(tag="TI-101", instrument_type="Temp Ind", function_code="TI", loop_number="101", confidence=None)
        rec = DrawingAnalysisRecord(
            drawing_id="TEST-004",
            source_path="test.png",
            drawing_type=DrawingType.PID,
            title_block=TitleBlockInfo(),
            equipment=(eq,),
            instruments=(inst,),
        )
        chunks = export_for_rag(rec)
        assert len(chunks) >= 2
        eq_chunk = next(c for c in chunks if c.metadata.chunk_strategy == "equipment_list")
        assert "conf: null" in eq_chunk.content

