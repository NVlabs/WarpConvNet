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
    """__enter__ and __exit__ to time a block of code.
    Returns the elapsed time in milliseconds.
    """

    def __init__(self):
        self.start_event = None
        self.end_event = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        return self.elapsed_time
