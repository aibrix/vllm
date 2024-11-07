#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <iostream>
#include <cstdlib>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

torch::Tensor flat_gemm(const torch::Tensor& x, const torch::Tensor& weight,
                        const std::optional<torch::Tensor>& bias) {
  // Check that inputs are compatible
  TORCH_CHECK(x.size(1) == weight.size(1), "Incompatible matrix dimensions for GEMM.");

  // Update Warpshape M to be 8 for handling flat gemm with Warpshape<M=8, N=64, K=8>
  using ThreadblockShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using WarpShape = cutlass::gemm::GemmShape<8, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  // Epilogue output operation
  // using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
  //   cutlass::half_t, 1, cutlass::half_t, cutlass::half_t,
  //   cutlass::epilogue::thread::ScaleType::Default,
  //   cutlass::FloatRoundStyle::round_to_nearest
  // >;


  // CUTLASS GEMM Configuration
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,                   // Data type of A (x)
      cutlass::layout::RowMajor,         // Layout of A
      cutlass::half_t,                   // Data type of B (weight)
      cutlass::layout::ColumnMajor,      // Layout of B
      cutlass::half_t,                   // Data type of C (output)
      cutlass::layout::RowMajor,         // Layout of C
      cutlass::half_t,                   // Data type for accumulation
      cutlass::arch::OpClassSimt,        // Operator class (SIMT for scalar computation)
      cutlass::arch::Sm80,               // Target architecture (Volta, Turing, etc.)
      ThreadblockShape,                  // Threadblock shape
      WarpShape,                         // Warp shape
      InstructionShape                   // Instruction shape
  >;

//   // CUTLASS GEMM Configuration
//   using Gemm = cutlass::gemm::device::Gemm<
//       cutlass::half_t,                             // Data type of A (x)
//       cutlass::layout::RowMajor,         // Layout of A
//       cutlass::half_t,                             // Data type of B (weight)
//       cutlass::layout::RowMajor,         // Layout of B
//       cutlass::half_t,                             // Data type of C (output)
//       cutlass::layout::RowMajor,         // Layout of C
//       cutlass::half_t,                             // Data type for accumulation
//       cutlass::arch::OpClassSimt,        // Operator class (SIMT for scalar computation)
//       cutlass::arch::Sm80,               // Target architecture (Volta, Turing, etc.)
//       ThreadblockShape,                  // Threadblock shape
//       WarpShape,                         // Warp shape
//       InstructionShape,                  // Instruction shape
//       EpilogueOutputOp,                  // Epilogue output operation
//       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, // Threadblock swizzle
//       2,                                 // Stages
//       1,                                 // AlignmentA
//       1,                                 // AlignmentB
//       false,                             // SplitKSerial
//       cutlass::arch::OpMultiplyAdd,       // Operator
//       false,                             // GatherA
//       false,                             // GatherB
//       false,                             // ScatterD
//       cutlass::layout::NoPermute          // PermuteDLayout
//   >;

  // Matrix dimensions
  int M = x.size(0);  // Rows of x
  int K = x.size(1);  // Columns of x, which must match rows of weight
  int N = weight.size(0);  // Columns of weight, which is also the columns of output

  // Allocate output tensor on CUDA
  torch::Tensor output = torch::empty({M, N}, x.options());

  // Get raw pointers for input and output tensors  
  torch::Half* d_x = x.data_ptr<torch::Half>();
  torch::Half* d_weight = weight.data_ptr<torch::Half>();
  torch::Half* d_output = output.data_ptr<torch::Half>();

  // Create CUTLASS GEMM object
  Gemm gemm_op;
  // printf("ThreadblockShape kM: %d\n", Gemm::ThreadblockShape::kM);
  // printf("ThreadblockShape kN: %d\n", Gemm::ThreadblockShape::kN);
  // printf("ThreadblockShape kK: %d\n", Gemm::ThreadblockShape::kK);
  // printf("WarpShape kM: %d\n", Gemm::WarpShape::kM);
  // printf("WarpShape kN: %d\n", Gemm::WarpShape::kN);
  // printf("WarpShape kK: %d\n", Gemm::WarpShape::kK);
  // printf("InstructionShape kM: %d\n", Gemm::InstructionShape::kM);
  // printf("InstructionShape kN: %d\n", Gemm::InstructionShape::kN);
  // printf("InstructionShape kK: %d\n", Gemm::InstructionShape::kK);

  // Set up the problem size and leading dimensions
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::half_t* d_x_cutlass = reinterpret_cast<cutlass::half_t*>(d_x);
  cutlass::half_t* d_weight_cutlass = reinterpret_cast<cutlass::half_t*>(d_weight);
  cutlass::half_t* d_output_cutlass = reinterpret_cast<cutlass::half_t*>(d_output);

  // Create GEMM arguments
  typename Gemm::Arguments arguments{
      problem_size,                    // Problem size (M, N, K)
      {d_x_cutlass, K},                // Tensor A (x) with leading dimension K
      {d_weight_cutlass, K},           // Tensor B (weight) with leading dimension K
      {d_output_cutlass, N},           // Tensor C (output) with leading dimension N
      {d_output_cutlass, N},           // Output tensor C (to be overwritten)
      {cutlass::half_t(1.0), cutlass::half_t(0.0)}             // Alpha = 1.0, Beta = 0.0 for the operation
  };

  // Run the CUTLASS GEMM operation
  cutlass::Status status = gemm_op(arguments);

  // Check for success
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM operation failed.");

  // If a bias is provided, add it to the output
  if (bias.has_value()) {
      output.add_(bias.value().to(torch::kHalf));
  }

  return output;  // Return the result tensor
}
