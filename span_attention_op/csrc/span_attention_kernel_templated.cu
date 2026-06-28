#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================
// Templated Optimized Span Attention Kernel
// Supports: C in {16, 24, 32, 48, 52, 56, 64}
// Note: 32, 48 use hand-optimized version (faster)
// ============================================

template <int C>
__global__ void span_attention_templated_fp32_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __shared__ float weight_shared[C * C];
    __shared__ float bias_shared[C];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    constexpr int WEIGHT_SIZE = C * C;
    for (int i = tid; i < WEIGHT_SIZE; i += blockDim.x) {
        weight_shared[i] = weight[i];
    }
    if (tid < C) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();
    
    float f3_local[C], x_local[C];
    
    if (valid) {
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            int idx = c * spatial_stride + spatial_offset;
            f3_local[c] = feat_deep[idx];
            x_local[c] = feat_low[idx];
        }
    }
    
    if (valid) {
        #pragma unroll
        for (int c_out = 0; c_out < C; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll
            for (int c_in = 0; c_in < C; c_in += 4) {
                guidance += weight_shared[c_out * C + c_in + 0] * f3_local[c_in + 0]
                          + weight_shared[c_out * C + c_in + 1] * f3_local[c_in + 1]
                          + weight_shared[c_out * C + c_in + 2] * f3_local[c_in + 2]
                          + weight_shared[c_out * C + c_in + 3] * f3_local[c_in + 3];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x_local[c_out] + f3_local[c_out]) * guidance;
        }
    }
}

template <int C>
__global__ void span_attention_templated_fp16_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int height, int width) {
    
    __shared__ half weight_shared[C * C];
    __shared__ half bias_shared[C];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    constexpr int WEIGHT_SIZE = C * C;
    for (int i = tid; i < WEIGHT_SIZE; i += blockDim.x) {
        weight_shared[i] = __half(weight[i]);
    }
    if (tid < C) {
        bias_shared[tid] = __half(bias[tid]);
    }
    __syncthreads();
    
    half f3_local[C], x_local[C];
    
    if (valid) {
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            int idx = c * spatial_stride + spatial_offset;
            f3_local[c] = __half(feat_deep[idx]);
            x_local[c] = __half(feat_low[idx]);
        }
    }
    
    if (valid) {
        #pragma unroll
        for (int c_out = 0; c_out < C; ++c_out) {
            half guidance = bias_shared[c_out];
            
            #pragma unroll
            for (int c_in = 0; c_in < C; c_in += 4) {
                guidance = __hfma(weight_shared[c_out * C + c_in + 0], f3_local[c_in + 0], guidance);
                guidance = __hfma(weight_shared[c_out * C + c_in + 1], f3_local[c_in + 1], guidance);
                guidance = __hfma(weight_shared[c_out * C + c_in + 2], f3_local[c_in + 2], guidance);
                guidance = __hfma(weight_shared[c_out * C + c_in + 3], f3_local[c_in + 3], guidance);
            }
            
            half sum = __hadd(x_local[c_out], f3_local[c_out]);
            output[c_out * spatial_stride + spatial_offset] = __half(__hmul(sum, guidance));
        }
    }
}

#define LAUNCH_KERNEL_TEMPLATED_FP32(C) \
    case C: \
        span_attention_templated_fp32_kernel<C><<<blocks, threads_per_block, \
            (C * C + C) * sizeof(float), stream>>>( \
            feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr, \
            output_ptr, height, width); \
        break;

#define LAUNCH_KERNEL_TEMPLATED_FP16(C) \
    case C: \
        span_attention_templated_fp16_kernel<C><<<blocks, threads_per_block, \
            (C * C + C) * sizeof(half), stream>>>( \
            feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr, \
            output_ptr, height, width); \
        break;

at::Tensor span_attention_forward_cuda_templated_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_low.size(0) == 1, "Inference only supports batch_size=1");
    TORCH_CHECK(feat_low.sizes() == feat_deep.sizes(), "Shape mismatch");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Float, "Input must be float32");

    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);

    if (height == 0 || width == 0 || channels == 0) {
        return output;
    }

    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const float* feat_low_ptr = feat_low.data_ptr<float>();
    const float* feat_deep_ptr = feat_deep.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int total_pixels = height * width;
    constexpr int threads_per_block = 256;
    const int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    
    const size_t required_shared_mem = (channels * channels + channels) * sizeof(float);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, feat_low.get_device());
    
    if (required_shared_mem > prop.sharedMemPerBlock) {
        TORCH_CHECK(false, 
            "Templated kernel requires ", required_shared_mem, 
            " bytes shared memory, but GPU only has ", prop.sharedMemPerBlock);
    }

    switch (channels) {
        LAUNCH_KERNEL_TEMPLATED_FP32(16)
        LAUNCH_KERNEL_TEMPLATED_FP32(24)
        LAUNCH_KERNEL_TEMPLATED_FP32(52)
        LAUNCH_KERNEL_TEMPLATED_FP32(56)
        LAUNCH_KERNEL_TEMPLATED_FP32(64)
        default:
            TORCH_CHECK(false, "Templated FP32: Only 16, 24, 52, 56, 64 channels supported. "
                        "Use 32 or 48 for hand-optimized kernel, or other channels for general.");
            break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor span_attention_forward_cuda_templated_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_low.size(0) == 1, "Inference only supports batch_size=1");
    TORCH_CHECK(feat_low.sizes() == feat_deep.sizes(), "Shape mismatch");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Half, "Input must be float16");

    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);

    if (height == 0 || width == 0 || channels == 0) {
        return output;
    }

    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const at::Half* feat_low_ptr = feat_low.data_ptr<at::Half>();
    const at::Half* feat_deep_ptr = feat_deep.data_ptr<at::Half>();
    const at::Half* weight_ptr = weight.data_ptr<at::Half>();
    const at::Half* bias_ptr = bias.data_ptr<at::Half>();
    at::Half* output_ptr = output.data_ptr<at::Half>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int total_pixels = height * width;
    constexpr int threads_per_block = 256;
    const int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    
    const size_t required_shared_mem = (channels * channels + channels) * sizeof(half);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, feat_low.get_device());
    
    if (required_shared_mem > prop.sharedMemPerBlock) {
        TORCH_CHECK(false, 
            "Templated FP16 kernel requires ", required_shared_mem, 
            " bytes shared memory, but GPU only has ", prop.sharedMemPerBlock);
    }

    switch (channels) {
        LAUNCH_KERNEL_TEMPLATED_FP16(16)
        LAUNCH_KERNEL_TEMPLATED_FP16(24)
        LAUNCH_KERNEL_TEMPLATED_FP16(52)
        LAUNCH_KERNEL_TEMPLATED_FP16(56)
        LAUNCH_KERNEL_TEMPLATED_FP16(64)
        default:
            TORCH_CHECK(false, "Templated FP16: Only 16, 24, 52, 56, 64 channels supported.");
            break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

#undef LAUNCH_KERNEL_TEMPLATED_FP32
#undef LAUNCH_KERNEL_TEMPLATED_FP16
