# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from warpconvnet.models.trellis2.pipeline import pipeline_config_from_json_args


def _pipeline_args():
    return {
        "sparse_structure_sampler": {
            "params": {
                "steps": 12,
                "guidance_strength": 7.5,
                "guidance_rescale": 0.7,
                "guidance_interval": [0.6, 1.0],
                "rescale_t": 5.0,
            }
        },
        "shape_slat_sampler": {
            "params": {
                "steps": 16,
                "guidance_strength": 8.0,
                "guidance_rescale": 0.5,
                "guidance_interval": [0.5, 1.0],
                "rescale_t": 3.0,
            }
        },
    }


def test_pipeline_config_from_json_args_uses_published_defaults():
    cfg = pipeline_config_from_json_args(_pipeline_args())
    assert cfg.ss_steps == 12
    assert cfg.slat_steps == 16
    assert cfg.ss_guidance_interval == (0.6, 1.0)
    assert cfg.slat_guidance_strength == 8.0


def test_pipeline_config_from_json_args_applies_step_overrides():
    cfg = pipeline_config_from_json_args(_pipeline_args(), steps=3, slat_steps=5)
    assert cfg.ss_steps == 3
    assert cfg.slat_steps == 5
