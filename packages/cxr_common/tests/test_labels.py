import pytest

from cxr_common.labels import (
    CHEXPERT_MAP,
    DESCRIPTIONS,
    DISPLAY_NAMES,
    NIH_LABELS,
    VINDR_MAP,
    parse_nih_finding_string,
)


def test_parse_findings() -> None:
    assert sum(parse_nih_finding_string("No Finding")) == 0
    v = parse_nih_finding_string("Effusion|Mass")
    assert v[NIH_LABELS.index("Effusion")] == 1 and v[NIH_LABELS.index("Mass")] == 1
    assert sum(v) == 2
    with pytest.raises(ValueError, match="unknown"):
        parse_nih_finding_string("Fracture")


def test_every_label_has_text() -> None:
    assert set(DISPLAY_NAMES) == set(NIH_LABELS) == set(DESCRIPTIONS)


def test_mappings_target_known_outputs() -> None:
    for m in (*CHEXPERT_MAP, *VINDR_MAP):
        assert all(n in NIH_LABELS for n in m.nih)
        assert m.approximate == (len(m.nih) > 1)
