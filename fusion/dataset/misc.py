from enum import auto
from strenum import LowercaseStrEnum


class SetId(LowercaseStrEnum):
    TRAIN: "SetId" = auto()  # type: ignore
    TEST: "SetId" = auto()  # type: ignore
    VALID: "SetId" = auto()  # type: ignore
    INFER: "SetId" = auto()  # type: ignore
