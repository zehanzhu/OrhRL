from __future__ import annotations

ROLE_SHARING = "role_sharing"
ROLE_SPECIFIC = "role_specific"

SUPPORTED_SPECIALIZATION_MODES = (
    ROLE_SHARING,
    ROLE_SPECIFIC,
)

def validate_specialization_mode(value) -> str:
    mode = str(value)
    if mode in SUPPORTED_SPECIALIZATION_MODES:
        return mode

    raise ValueError(
        f"Unsupported specialization '{mode}'. "
        f"Supported values: {', '.join(SUPPORTED_SPECIALIZATION_MODES)}"
    )
