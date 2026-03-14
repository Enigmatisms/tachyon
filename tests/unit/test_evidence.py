"""Unit tests for evidence chain dataclasses."""
from tachyon.models.evidence import (
    EvidenceChain,
    MetricEvidence,
    SassEvidence,
    SourceEvidence,
)


class TestMetricEvidence:
    def test_creation(self):
        me = MetricEvidence(
            name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
            value=85.0,
            threshold=60.0,
            status="HIGH",
            unit="%",
        )
        assert me.name == "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        assert me.value == 85.0
        assert me.threshold == 60.0
        assert me.status == "HIGH"
        assert me.unit == "%"

    def test_default_unit(self):
        me = MetricEvidence(name="m", value=1.0, threshold=0.5, status="OK")
        assert me.unit == ""


class TestSourceEvidence:
    def test_creation(self):
        se = SourceEvidence(
            file="kernel.cu", line=42, snippet="float x = a[i];", function="myKernel"
        )
        assert se.file == "kernel.cu"
        assert se.line == 42
        assert se.snippet == "float x = a[i];"
        assert se.function == "myKernel"

    def test_no_function(self):
        se = SourceEvidence(file="a.cu", line=1, snippet="code")
        assert se.function is None


class TestSassEvidence:
    def test_creation(self):
        se = SassEvidence(
            pc=0xDEAD, instruction="LDG.E R2, [R4]", note="Global load", ptx="ld.global.f32"
        )
        assert se.pc == 0xDEAD
        assert se.instruction == "LDG.E R2, [R4]"
        assert se.note == "Global load"
        assert se.ptx == "ld.global.f32"

    def test_no_ptx(self):
        se = SassEvidence(pc=0x1000, instruction="FADD", note="")
        assert se.ptx is None


class TestEvidenceChain:
    def test_defaults(self):
        ec = EvidenceChain()
        assert ec.metric_evidence == []
        assert ec.source_evidence is None
        assert ec.sass_evidence is None

    def test_full_chain(self):
        me = MetricEvidence(name="m", value=90.0, threshold=60.0, status="HIGH")
        se = SourceEvidence(file="k.cu", line=10, snippet="x = a[i];")
        sa = SassEvidence(pc=0x100, instruction="LDG.E R2, [R4]", note="global load")
        ec = EvidenceChain(
            metric_evidence=[me], source_evidence=se, sass_evidence=sa
        )
        assert len(ec.metric_evidence) == 1
        assert ec.source_evidence.file == "k.cu"
        assert ec.sass_evidence.pc == 0x100

    def test_multiple_metric_evidence(self):
        m1 = MetricEvidence(name="a", value=1.0, threshold=0.5, status="HIGH")
        m2 = MetricEvidence(name="b", value=2.0, threshold=1.0, status="HIGH")
        ec = EvidenceChain(metric_evidence=[m1, m2])
        assert len(ec.metric_evidence) == 2
        assert ec.metric_evidence[0].name == "a"
        assert ec.metric_evidence[1].name == "b"
