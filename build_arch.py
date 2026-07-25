# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact CUDA architecture handling for the extension build.

PyTorch's architecture parser changes with the installed PyTorch release and
has historically expanded named architecture families (``Ampere``,
``Blackwell``) and inferred forward-compatible cubins the caller never asked
for.  WarpConvNet keeps the build contract deliberately small: every requested
token maps to exactly one cubin target.  A ``+PTX`` suffix adds PTX only when
it was explicitly requested; it never adds a second cubin, and no architecture
is ever inferred from another one being present.

This module intentionally has no torch dependency.  ``setup.py`` loads it by
file path (without importing any package ``__init__``), so the parsing and
validation logic is covered by CPU-only tests that need neither a GPU nor a
CUDA-enabled PyTorch build.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence


class CudaArchError(ValueError):
    """Raised when an exact CUDA build target cannot be honored."""


# Certified architecture catalog. A build target's base code (major*10+minor)
# must appear here or the request is rejected. SM103 and SM121 are left
# uncertified on purpose: they are not aliased onto SM100/SM120 cubins.
CERTIFIED_ARCH_CODES = frozenset({70, 75, 80, 86, 87, 89, 90, 100, 120})


@dataclass(frozen=True)
class CudaArchTarget:
    """One exact CUDA architecture requested by the build."""

    major: int
    minor: int
    accelerated: bool = False
    emit_ptx: bool = False

    @property
    def base_code(self) -> int:
        """Integer compute capability, for example ``120`` for ``12.0``."""

        return self.major * 10 + self.minor

    @property
    def code(self) -> str:
        """NVCC architecture suffix, for example ``100a`` or ``120``."""

        suffix = "a" if self.accelerated else ""
        return f"{self.base_code}{suffix}"

    @property
    def canonical_token(self) -> str:
        """Canonical ``TORCH_CUDA_ARCH_LIST`` spelling of this target."""

        suffix = "a" if self.accelerated else ""
        ptx = "+PTX" if self.emit_ptx else ""
        return f"{self.major}.{self.minor}{suffix}{ptx}"

    @property
    def cubin_flag(self) -> str:
        """NVCC flag that emits this target's exact cubin (SASS)."""

        return f"-gencode=arch=compute_{self.code},code=sm_{self.code}"

    @property
    def ptx_flag(self) -> str:
        """NVCC flag that emits PTX for this target (only when requested)."""

        return f"-gencode=arch=compute_{self.code},code=compute_{self.code}"


_DOTTED_ARCH_RE = re.compile(
    r"^(?P<major>\d{1,2})\.(?P<minor>\d)(?P<accelerated>a)?$",
    re.IGNORECASE,
)
_COMPACT_ARCH_RE = re.compile(
    r"^(?P<code>\d{2,3})(?P<accelerated>a)?$",
    re.IGNORECASE,
)


def parse_cuda_arch_token(token: str) -> CudaArchTarget:
    """Parse one exact CUDA architecture token.

    Accepted spellings include dotted (``12.0``, ``10.0a``), compact
    (``120``, ``100a``), and ``sm_``/``compute_`` prefixed forms
    (``sm_100a``, ``compute_120``), each optionally carrying a trailing
    ``+PTX`` (case-insensitive) that additionally emits PTX for that target.

    The trailing ``a`` is an identity marker for an accelerated target; it is
    never inferred.  CUDA family names such as ``Blackwell`` are rejected
    because they would expand to more than one exact target, and any base code
    outside ``CERTIFIED_ARCH_CODES`` is rejected by name.
    """

    original = token
    text = token.strip().lower()
    emit_ptx = text.endswith("+ptx")
    if emit_ptx:
        text = text[:-4].strip()

    for prefix in ("sm_", "compute_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    dotted = _DOTTED_ARCH_RE.fullmatch(text)
    if dotted:
        major = int(dotted.group("major"))
        minor = int(dotted.group("minor"))
        accelerated = dotted.group("accelerated") is not None
    else:
        compact = _COMPACT_ARCH_RE.fullmatch(text)
        if not compact:
            raise CudaArchError(
                f"invalid CUDA architecture token {original!r}; use an exact "
                "target such as '8.0', '10.0a', 'sm_100a', or '12.0' "
                "(named families like 'Blackwell' are not supported)"
            )
        compact_code = compact.group("code")
        major = int(compact_code[:-1])
        minor = int(compact_code[-1])
        accelerated = compact.group("accelerated") is not None

    target = CudaArchTarget(
        major=major,
        minor=minor,
        accelerated=accelerated,
        emit_ptx=emit_ptx,
    )

    if target.base_code not in CERTIFIED_ARCH_CODES:
        known = ", ".join(str(code) for code in sorted(CERTIFIED_ARCH_CODES))
        raise CudaArchError(
            f"uncertified CUDA architecture token {original!r} "
            f"(base code {target.base_code}); certified base codes: {known}"
        )

    return target


def parse_cuda_arch_list(value: str) -> tuple[CudaArchTarget, ...]:
    """Parse an exact, ordered, de-duplicated CUDA architecture list.

    Tokens are split on commas, semicolons, and whitespace; request order is
    preserved.  Duplicates keyed on ``(major, minor, accelerated)`` collapse to
    one entry, and a repeated token carrying ``+PTX`` upgrades ``emit_ptx`` on
    the existing entry rather than adding a second cubin.
    """

    tokens = value.replace(",", " ").replace(";", " ").split()
    if not tokens:
        raise CudaArchError("CUDA architecture list is empty")

    ordered: list[CudaArchTarget] = []
    positions: dict[tuple[int, int, bool], int] = {}
    for token in tokens:
        target = parse_cuda_arch_token(token)
        key = (target.major, target.minor, target.accelerated)
        previous_position = positions.get(key)
        if previous_position is None:
            positions[key] = len(ordered)
            ordered.append(target)
        elif target.emit_ptx and not ordered[previous_position].emit_ptx:
            # A repeated "+PTX" request strengthens the same exact target
            # without emitting a duplicate cubin.
            ordered[previous_position] = replace(ordered[previous_position], emit_ptx=True)
    return tuple(ordered)


def cuda_gencode_flags(targets: Sequence[CudaArchTarget]) -> list[str]:
    """Return exact NVCC gencode flags for ``targets``.

    One cubin flag is emitted per target in request order; a PTX flag is
    emitted only for a target carrying an explicit ``+PTX`` suffix.  Nothing
    else is added.
    """

    flags: list[str] = []
    for target in targets:
        flags.append(target.cubin_flag)
        if target.emit_ptx:
            flags.append(target.ptx_flag)
    return flags


def cuda_feature_macros(targets: Sequence[CudaArchTarget]) -> list[str]:
    """Return the ``-DWARPCONVNET_SM*_ENABLED`` macros implied by ``targets``.

    - ``WARPCONVNET_SM80_ENABLED`` when any target's base code is >= 80. This
      gates generic ``mma.sync`` tensor-core source and is safe to enable as
      long as no incompatible cubin is injected.
    - ``WARPCONVNET_SM90_ENABLED`` only when an accelerated ``9.0a`` target is
      explicitly present. The SM90 backends are WGMMA and require ``sm_90a``.
    - ``WARPCONVNET_SM100_ENABLED`` only when an accelerated ``10.0a`` target
      is explicitly present.
    """

    macros: list[str] = []
    if any(target.base_code >= 80 for target in targets):
        macros.append("-DWARPCONVNET_SM80_ENABLED=1")
    if any(target.base_code == 90 and target.accelerated for target in targets):
        macros.append("-DWARPCONVNET_SM90_ENABLED=1")
    if any(target.base_code == 100 and target.accelerated for target in targets):
        macros.append("-DWARPCONVNET_SM100_ENABLED=1")
    return macros
