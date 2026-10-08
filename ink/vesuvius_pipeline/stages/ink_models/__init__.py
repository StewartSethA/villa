"""Trimmed ink_models package (vendored): only the families Route B runs by default."""
from __future__ import annotations
from .registry import INK_BY_NAME, INK_FAMILIES, InkFamily, ink_family  # noqa: F401
from . import ink9um_student, grandprize_dense, reader_v2_dense  # noqa: F401
FAMILIES = {"ink9um_student": ink9um_student, "grandprize_dense": grandprize_dense, "reader_v2_dense": reader_v2_dense}
