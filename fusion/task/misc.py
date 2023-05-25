from enum import auto
from strenum import LowercaseStrEnum


class TaskId(LowercaseStrEnum):
    PRETRAINING: "TaskId" = auto()
    LINEAR_EVALUATION: "TaskId" = auto()
    LOGREG_EVALUATION: "TaskId" = auto()
    SALIENCY: "TaskId" = auto()
    TSNE: "TaskId" = auto()
    FEATURE_EXTRACTION: "TaskId" = auto()
    INFERENCE: "TaskId" = auto()
    LOGITS_EXTRACTION: "TaskId" = auto()
