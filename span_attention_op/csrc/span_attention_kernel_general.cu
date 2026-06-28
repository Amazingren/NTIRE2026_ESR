#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================
// 通用 Span Attention Kernel - 支持任意通道数和 batch size
// ============================================
// 策略：
// - 保持原有 32/48 通道的高度优化版本
// - 对其他通道数使用通用实现
// - 支持任意 batch size 通过 batch loop 实现

// -------------------- FP32 Global Memory Version (fallback) --------------------
// 当 shared memory 不够时使用，直接从 global memory 读取 weight
template <int BLOCK_SIZE = 256>
__global__ void span_attention_global_fp32_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels, int height, int width) {
    
    const int tid = threadIdx.x;
    const int total_spatial = height * width;
    
    const int nhw_idx = blockIdx.x * BLOCK_SIZE + tid;
    const int n = nhw_idx / total_spatial;
    const int hw = nhw_idx % total_spatial;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (n < batch_size && h < height && w < width);
    const int spatial_offset = h * width + w;
    
    // 共享内存：只放 bias (channels 个 float，通常很小)
    extern __shared__ float bias_shared[];
    
    // 协作加载 bias
    for (int i = tid; i < channels; i += BLOCK_SIZE) {
        bias_shared[i] = bias[i];
    }
    __syncthreads();
    
    if (valid) {
        const int base_idx = n * channels * total_spatial + spatial_offset;
        
        for (int c_out = 0; c_out < channels; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll 4
            for (int c_in = 0; c_in < channels; ++c_in) {
                int feat_idx = base_idx + c_in * total_spatial;
                // 直接从 global memory 读取 weight
                guidance += weight[c_out * channels + c_in] * feat_deep[feat_idx];
            }
            
            int out_idx = base_idx + c_out * total_spatial;
            float x_val = feat_low[out_idx];
            float f3_val = feat_deep[out_idx];
            output[out_idx] = (x_val + f3_val) * guidance;
        }
    }
}

// -------------------- FP32 General Version (with shared weight) --------------------
template <int BLOCK_SIZE = 256>
__global__ void span_attention_shared_fp32_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels, int height, int width) {
    
    const int tid = threadIdx.x;
    const int total_spatial = height * width;
    
    const int nhw_idx = blockIdx.x * BLOCK_SIZE + tid;
    const int n = nhw_idx / total_spatial;
    const int hw = nhw_idx % total_spatial;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (n < batch_size && h < height && w < width);
    const int spatial_offset = h * width + w;
    
    // 共享内存：weight [channels, channels] + bias [channels]
    extern __shared__ float shared_mem[];
    float* weight_shared = shared_mem;
    float* bias_shared = shared_mem + channels * channels;
    
    // 协作加载 weight
    const int weight_size = channels * channels;
    for (int i = tid; i < weight_size; i += BLOCK_SIZE) {
        weight_shared[i] = weight[i];
    }
    // 协作加载 bias
    for (int i = tid; i < channels; i += BLOCK_SIZE) {
        bias_shared[i] = bias[i];
    }
    __syncthreads();
    
    if (valid) {
        const int base_idx = n * channels * total_spatial + spatial_offset;
        
        for (int c_out = 0; c_out < channels; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll 4
            for (int c_in = 0; c_in < channels; ++c_in) {
                int feat_idx = base_idx + c_in * total_spatial;
                guidance += weight_shared[c_out * channels + c_in] * feat_deep[feat_idx];
            }
            
            int out_idx = base_idx + c_out * total_spatial;
            float x_val = feat_low[out_idx];
            float f3_val = feat_deep[out_idx];
            output[out_idx] = (x_val + f3_val) * guidance;
        }
    }
}

// -------------------- FP16 General Version --------------------
template <int BLOCK_SIZE = 256>
__global__ void span_attention_general_fp16_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int batch_size, int channels, int height, int width) {
    
    const int tid = threadIdx.x;
    const int total_spatial = height * width;
    
    const int nhw_idx = blockIdx.x * BLOCK_SIZE + tid;
    const int n = nhw_idx / total_spatial;
    const int hw = nhw_idx % total_spatial;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (n < batch_size && h < height && w < width);
    const int spatial_offset = h * width + w;
    
    extern __shared__ half shared_mem_half[];
    half* weight_shared = shared_mem_half;
    half* bias_shared = shared_mem_half + channels * channels;
    
    const int weight_size = channels * channels;
    for (int i = tid; i < weight_size; i += BLOCK_SIZE) {
        weight_shared[i] = __half(weight[i]);
    }
    for (int i = tid; i < channels; i += BLOCK_SIZE) {
        bias_shared[i] = __half(bias[i]);
    }
    __syncthreads();
    
    if (valid) {
        const int base_idx = n * channels * total_spatial + spatial_offset;
        
        for (int c_out = 0; c_out < channels; ++c_out) {
            half guidance = bias_shared[c_out];
            
            #pragma unroll 4
            for (int c_in = 0; c_in < channels; ++c_in) {
                int feat_idx = base_idx + c_in * total_spatial;
                guidance = __hfma(weight_shared[c_out * channels + c_in], 
                                 __half(feat_deep[feat_idx]), guidance);
            }
            
            int out_idx = base_idx + c_out * total_spatial;
            half x_val = __half(feat_low[out_idx]);
            half f3_val = __half(feat_deep[out_idx]);
            half sum = __hadd(x_val, f3_val);
            output[out_idx] = __half(__hmul(sum, guidance));
        }
    }
}

// ============================================
// PyTorch 接口 - 通用版本
// ============================================

// 通用 FP32 前向传播
at::Tensor span_attention_forward_cuda_general_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_low.sizes() == feat_deep.sizes(), "Shape mismatch");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Float, "Input must be float32");

    const int batch_size = feat_low.size(0);
    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);

    if (height == 0 || width == 0 || channels == 0 || batch_size == 0) {
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

    constexpr int BLOCK_SIZE = 256;
    const int total_elements = batch_size * height * width;
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 共享内存大小：weight[C*C] + bias[C]
    const size_t shared_weight_size = (channels * channels + channels) * sizeof(float);
    // 共享内存大小：只放 bias
    const size_t shared_bias_size = channels * sizeof(float);
    
    // 检查 GPU 共享内存限制
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, feat_low.get_device());
    
    if (shared_weight_size <= prop.sharedMemPerBlock) {
        // 共享内存足够，使用 shared weight 版本
        span_attention_shared_fp32_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 
            shared_weight_size, stream>>>(
            feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
            output_ptr, batch_size, channels, height, width);
    } else {
        // 共享内存不足，使用 global memory 版本
        span_attention_global_fp32_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 
            shared_bias_size, stream>>>(
            feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
            output_ptr, batch_size, channels, height, width);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// 通用 FP16 前向传播
at::Tensor span_attention_forward_cuda_general_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_low.sizes() == feat_deep.sizes(), "Shape mismatch");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Half, "Input must be float16");

    const int batch_size = feat_low.size(0);
    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);

    if (height == 0 || width == 0 || channels == 0 || batch_size == 0) {
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

    constexpr int BLOCK_SIZE = 256;
    const int total_elements = batch_size * height * width;
    const int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    const size_t shared_mem_size = (channels * channels + channels) * sizeof(half);
    
    span_attention_general_fp16_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 
        shared_mem_size, stream>>>(
        feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
        output_ptr, batch_size, channels, height, width);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
