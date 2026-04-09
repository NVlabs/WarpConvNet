#!/usr/bin/env bash
# Check that source files contain the SPDX license header.
# Called by pre-commit with a list of staged file paths as arguments.

SPDX_TAG="SPDX-License-Identifier"
failed=0

for f in "$@"; do
    # Read first 5 lines — header should appear near the top
    if ! head -5 "$f" | grep -q "$SPDX_TAG"; then
        echo "Missing SPDX header: $f"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo ""
    echo "Add the appropriate header to the top of each file:"
    echo "  Python:  # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    echo "           # SPDX-License-Identifier: Apache-2.0"
    echo "  C/C++:   // SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
    echo "           // SPDX-License-Identifier: Apache-2.0"
    exit 1
fi
