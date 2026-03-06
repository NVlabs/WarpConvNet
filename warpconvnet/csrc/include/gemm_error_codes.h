// Return codes for GEMM operations
#pragma once

namespace warpconvnet {
namespace gemm {

enum class GemmStatus {
  kSuccess = 0,
  kErrorProblemNotSupported = -1,
  kErrorKernelInitialization = -2,
  kErrorKernelExecution = -3,
  kErrorUnsupportedConfig = -4,
  kErrorInvalidParameters = -5,
  kErrorMixedInputUnsupported = -6,
};

inline const char *GemmStatusToString(GemmStatus status) {
  switch (status) {
    case GemmStatus::kSuccess:
      return "Success";
    case GemmStatus::kErrorProblemNotSupported:
      return "Problem size not supported";
    case GemmStatus::kErrorKernelInitialization:
      return "Kernel initialization failed";
    case GemmStatus::kErrorKernelExecution:
      return "Kernel execution failed";
    case GemmStatus::kErrorUnsupportedConfig:
      return "Unsupported precision/configuration";
    case GemmStatus::kErrorInvalidParameters:
      return "Invalid parameters";
    case GemmStatus::kErrorMixedInputUnsupported:
      return "Mixed input precision unsupported";
    default:
      return "Unknown error";
  }
}

}  // namespace gemm
}  // namespace warpconvnet
