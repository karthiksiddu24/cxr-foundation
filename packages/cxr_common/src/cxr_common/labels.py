"""Finding definitions and cross-dataset label harmonization.

The model predicts the 14 NIH ChestX-ray14 findings. Cross-dataset evaluation (CheXpert, VinDr)
uses only findings with a defensible mapping. Approximate mappings are flagged so reports can show
them separately. See docs/labels.md for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

NIH_LABELS: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)
NUM_CLASSES: Final[int] = len(NIH_LABELS)

# Display names (UI, reports).
DISPLAY_NAMES: Final[dict[str, str]] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural effusion",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Pleural_Thickening": "Pleural thickening",
    "Hernia": "Hernia",
}

# Plain-language descriptions shown in UI tooltips. Educational, not diagnostic.
DESCRIPTIONS: Final[dict[str, str]] = {
    "Atelectasis": "Partial collapse or incomplete inflation of part of a lung.",
    "Cardiomegaly": "The heart appears larger than expected relative to the chest.",
    "Effusion": "Fluid collecting in the space between the lung and the chest wall.",
    "Infiltration": "A hazy region in the lung that may reflect fluid, cells or inflammation. "
    "This label is broad and inconsistently used in the training data.",
    "Mass": "A spot in the lung larger than about 3 cm.",
    "Nodule": "A small, roughly round spot in the lung, usually under 3 cm.",
    "Pneumonia": "Lung infection. On an X-ray it often looks like consolidation; "
    "it cannot be confirmed from an image alone.",
    "Pneumothorax": "Air in the space around the lung, which can cause the lung to collapse.",
    "Consolidation": "Part of the lung filled with fluid or tissue instead of air, "
    "appearing as a solid white area.",
    "Edema": "Excess fluid in the lung tissue, often related to heart function.",
    "Emphysema": "Damaged air sacs that make the lungs over-inflated, often linked to smoking.",
    "Fibrosis": "Scarring and thickening of lung tissue.",
    "Pleural_Thickening": "Thickening of the lining around the lungs, often from old inflammation.",
    "Hernia": "Abdominal contents pushing up into the chest, usually through the diaphragm.",
}


@dataclass(frozen=True)
class Mapping:
    """How a source-dataset label maps onto an NIH finding.

    `nih` lists the NIH output(s) whose probabilities are combined (max) to score the target label.
    """

    target: str
    nih: tuple[str, ...]
    approximate: bool = False


# CheXpert label -> NIH outputs. CheXpert's valid/test sets are radiologist-labeled and binary,
# so no uncertainty policy is needed for evaluation (training labels are not used).
CHEXPERT_MAP: Final[tuple[Mapping, ...]] = (
    Mapping("Atelectasis", ("Atelectasis",)),
    Mapping("Cardiomegaly", ("Cardiomegaly",)),
    Mapping("Consolidation", ("Consolidation",)),
    Mapping("Edema", ("Edema",)),
    Mapping("Pleural Effusion", ("Effusion",)),
    Mapping("Pneumonia", ("Pneumonia",)),
    Mapping("Pneumothorax", ("Pneumothorax",)),
    Mapping("Lung Lesion", ("Mass", "Nodule"), approximate=True),
)

# VinDr-CXR box class -> NIH output(s), for heatmap localization.
VINDR_MAP: Final[tuple[Mapping, ...]] = (
    Mapping("Atelectasis", ("Atelectasis",)),
    Mapping("Cardiomegaly", ("Cardiomegaly",)),
    Mapping("Consolidation", ("Consolidation",)),
    Mapping("Infiltration", ("Infiltration",)),
    Mapping("Nodule/Mass", ("Mass", "Nodule"), approximate=True),
    Mapping("Pleural effusion", ("Effusion",)),
    Mapping("Pleural thickening", ("Pleural_Thickening",)),
    Mapping("Pneumothorax", ("Pneumothorax",)),
    Mapping("Pulmonary fibrosis", ("Fibrosis",)),
)

# NIH BBox_List_2017 uses these names (note "Infiltrate").
NIH_BBOX_NAME_TO_LABEL: Final[dict[str, str]] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion",
    "Infiltrate": "Infiltration",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


def label_index(name: str) -> int:
    """Index of an NIH label in the model output vector."""
    return NIH_LABELS.index(name)


def parse_nih_finding_string(s: str) -> list[int]:
    """Parse NIH 'Finding Labels' (e.g. 'Effusion|Mass', 'No Finding') to a 14-dim 0/1 vector."""
    vec = [0] * NUM_CLASSES
    for part in s.split("|"):
        part = part.strip()
        if part in ("", "No Finding"):
            continue
        if part not in NIH_LABELS:
            raise ValueError(f"unknown NIH label: {part!r}")
        vec[NIH_LABELS.index(part)] = 1
    return vec
