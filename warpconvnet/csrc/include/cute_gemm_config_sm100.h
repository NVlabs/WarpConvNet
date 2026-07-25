// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//
// SM100-only compute-backend primitives for generated mask-GEMM kernels.
// Keep the existing cute_gemm_config.h contract unchanged.

#pragma once

#include <cstdint>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

namespace warpconvnet {
namespace cute_gemm {

struct Sm100UmmaBackendTag {};

// Logical front-end state shared by forward and dgrad native generators.
// Wgrad uses the same type with the maximum physical stride (12 words).
template <int TileM, int MaskWords>
struct MaskTileState {
  int real_rows[TileM];
  uint32_t mask_union[MaskWords];
};

template <class>
struct Sm100UmmaImplementationAvailable : cute::false_type {};

using Sm100TmemAllocator = cute::TMEM::Allocator1Sm;

template <int Stages, class AtomThreadShape>
using Sm100UmmaPipeline = cutlass::PipelineUmmaAsync<Stages, AtomThreadShape>;

}  // namespace cute_gemm
}  // namespace warpconvnet
