# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import torch
import numpy as np


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.min_time = np.inf

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        if self.elapsed < self.min_time:
            self.min_time = self.elapsed

    @property
    def elapsed(self):
        return self.end_time - self.start_time

    @property
    def min_elapsed(self):
        return self.min_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False


class CUDATimer:
    """Context manager for timing CUDA operations in milliseconds.

    Uses CUDA events for accurate GPU timing. Synchronizes before the
    start event to drain previous GPU work and prevent pipeline overlap
    from inflating measurements.

    Usage::

        timer = CUDATimer()
        with timer:
            kernel(...)
        print(timer.elapsed_time)  # milliseconds
    """

    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        # Drain any pending GPU work so the start event marks a clean
        # boundary. Without this, the measurement can include tail work
        # from a previous iteration.
        torch.cuda.current_stream().synchronize()
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        # Wait only for this event, not all device work. This avoids
        # blocking on unrelated streams/ranks during auto-tuning.
        self.end_event.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        return False
