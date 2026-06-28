#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================
// FP32 + FP16 双精度支持的 span attention CUDA kernel
// ============================================

// ==================== 48通道 FP32 版本 ====================
// 优化策略：float4向量化weight加载 + 寄存器缓存
__global__ void __launch_bounds__(256, 2) span_attention_48ch_shared_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ float weight_shared[48 * 48];
    __align__(16) __shared__ float bias_shared[48];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    // 向量化加载 weight 和 bias (这些是对齐的)
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* weight_shared_vec = reinterpret_cast<float4*>(weight_shared);
    
    #pragma unroll 4
    for (int i = tid; i < 576; i += 256) {
        weight_shared_vec[i] = weight_vec[i];
    }
    if (tid < 12) {
        reinterpret_cast<float4*>(bias_shared)[tid] = reinterpret_cast<const float4*>(bias)[tid];
    }
    __syncthreads();
    
    // 加载特征数据 (不使用向量化，因为可能不对齐)
    float f3[48], x[48];
    
    if (valid) {
        const float* feat_low_ptr = feat_low + spatial_offset;
        const float* feat_deep_ptr = feat_deep + spatial_offset;
        
        #pragma unroll 8
        for (int c = 0; c < 48; ++c) {
            int idx = c * spatial_stride;
            x[c] = feat_low_ptr[idx];
            f3[c] = feat_deep_ptr[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 6
        for (int c_out = 0; c_out < 48; ++c_out) {
            float guidance = bias_shared[c_out];
            const float* w_row = weight_shared + c_out * 48;
            
            #pragma unroll 6
            for (int i = 0; i < 6; ++i) {
                int base = i * 8;
                guidance += w_row[base + 0] * f3[base + 0]
                          + w_row[base + 1] * f3[base + 1]
                          + w_row[base + 2] * f3[base + 2]
                          + w_row[base + 3] * f3[base + 3]
                          + w_row[base + 4] * f3[base + 4]
                          + w_row[base + 5] * f3[base + 5]
                          + w_row[base + 6] * f3[base + 6]
                          + w_row[base + 7] * f3[base + 7];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x[c_out] + f3[c_out]) * guidance;
        }
    }
}

// ==================== 48通道 FP16 版本 ====================
// 优化策略：half2向量化weight加载 + HMMA
__global__ void __launch_bounds__(256, 2) span_attention_48ch_half_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ half weight_shared[48 * 48];
    __align__(16) __shared__ half bias_shared[48];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    // 向量化加载 weight 和 bias (这些是对齐的)
    const half2* weight_vec = reinterpret_cast<const half2*>(weight);
    half2* weight_shared_vec = reinterpret_cast<half2*>(weight_shared);
    
    #pragma unroll 4
    for (int i = tid; i < 1152; i += 256) {
        weight_shared_vec[i] = weight_vec[i];
    }
    if (tid < 24) {
        reinterpret_cast<half2*>(bias_shared)[tid] = reinterpret_cast<const half2*>(bias)[tid];
    }
    __syncthreads();
    
    // 加载特征数据 (不使用向量化，因为可能不对齐)
    half f3[48], x[48];
    
    if (valid) {
        const half* feat_low_ptr = reinterpret_cast<const half*>(feat_low) + spatial_offset;
        const half* feat_deep_ptr = reinterpret_cast<const half*>(feat_deep) + spatial_offset;
        
        #pragma unroll 12
        for (int c = 0; c < 48; ++c) {
            int idx = c * spatial_stride;
            x[c] = feat_low_ptr[idx];
            f3[c] = feat_deep_ptr[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 6
        for (int c_out = 0; c_out < 48; ++c_out) {
            half guidance = bias_shared[c_out];
            const half* w_row = weight_shared + c_out * 48;
            
            // 使用 half2 向量化乘加
            #pragma unroll 6
            for (int i = 0; i < 24; ++i) {
                half2 w = reinterpret_cast<const half2*>(w_row)[i];
                half2 f = reinterpret_cast<half2*>(f3)[i];
                guidance = __hfma(w.x, f.x, guidance);
                guidance = __hfma(w.y, f.y, guidance);
            }
            
            half sum = __hadd(x[c_out], f3[c_out]);
            output[c_out * spatial_stride + spatial_offset] = __hmul(sum, guidance);
        }
    }
}

// ==================== 32通道 FP32 版本 ====================
__global__ void __launch_bounds__(256, 4) span_attention_32ch_shared_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ float weight_shared[32 * 32];
    __align__(16) __shared__ float bias_shared[32];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    #pragma unroll 4
    for (int i = tid; i < 32 * 32; i += 256) {
        weight_shared[i] = weight[i];
    }
    if (tid < 32) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();
    
    float f3[32], x[32];
    
    if (valid) {
        const float* feat_low_ptr = feat_low + spatial_offset;
        const float* feat_deep_ptr = feat_deep + spatial_offset;
        
        #pragma unroll 8
        for (int c = 0; c < 32; ++c) {
            int idx = c * spatial_stride;
            x[c] = feat_low_ptr[idx];
            f3[c] = feat_deep_ptr[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 8
        for (int c_out = 0; c_out < 32; ++c_out) {
            float guidance = bias_shared[c_out];
            const float* w_row = weight_shared + c_out * 32;
            
            #pragma unroll 4
            for (int i = 0; i < 4; ++i) {
                int base = i * 8;
                guidance += w_row[base + 0] * f3[base + 0]
                          + w_row[base + 1] * f3[base + 1]
                          + w_row[base + 2] * f3[base + 2]
                          + w_row[base + 3] * f3[base + 3]
                          + w_row[base + 4] * f3[base + 4]
                          + w_row[base + 5] * f3[base + 5]
                          + w_row[base + 6] * f3[base + 6]
                          + w_row[base + 7] * f3[base + 7];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x[c_out] + f3[c_out]) * guidance;
        }
    }
}

// ==================== 32通道 FP16 版本 ====================
__global__ void __launch_bounds__(256, 4) span_attention_32ch_half_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ half weight_shared[32 * 32];
    __align__(16) __shared__ half bias_shared[32];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    #pragma unroll 8
    for (int i = tid; i < 32 * 32; i += 256) {
        weight_shared[i] = __half(weight[i]);
    }
    if (tid < 32) {
        bias_shared[tid] = __half(bias[tid]);
    }
    __syncthreads();
    
    half f3[32], x[32];
    
    if (valid) {
        const half* feat_low_ptr = reinterpret_cast<const half*>(feat_low) + spatial_offset;
        const half* feat_deep_ptr = reinterpret_cast<const half*>(feat_deep) + spatial_offset;
        
        #pragma unroll 8
        for (int c = 0; c < 32; ++c) {
            int idx = c * spatial_stride;
            x[c] = feat_low_ptr[idx];
            f3[c] = feat_deep_ptr[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 8
        for (int c_out = 0; c_out < 32; ++c_out) {
            half guidance = bias_shared[c_out];
            const half* w_row = weight_shared + c_out * 32;
            
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                guidance = __hfma(w_row[i], f3[i], guidance);
            }
            
            half sum = __hadd(x[c_out], f3[c_out]);
            output[c_out * spatial_stride + spatial_offset] = __hmul(sum, guidance);
        }
    }
}

// ==================== PyTorch FP32 接口 ====================
at::Tensor span_attention_forward_cuda_optimized(
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

    switch (channels) {
        case 48:
            span_attention_48ch_shared_kernel<<<blocks, threads_per_block, 
                (48 * 48 + 48) * sizeof(float), stream>>>(
                feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
                output_ptr, height, width);
            break;
        case 32:
            span_attention_32ch_shared_kernel<<<blocks, threads_per_block,
                (32 * 32 + 32) * sizeof(float), stream>>>(
                feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
                output_ptr, height, width);
            break;
        default:
            TORCH_CHECK(false, "FP32: Only 32 and 48 channels supported");
            break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ==================== PyTorch FP16 接口 ====================
at::Tensor span_attention_forward_cuda_half(
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

    switch (channels) {
        case 48:
            span_attention_48ch_half_kernel<<<blocks, threads_per_block, 
                (48 * 48 + 48) * sizeof(half), stream>>>(
                feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
                output_ptr, height, width);
            break;
        case 32:
            span_attention_32ch_half_kernel<<<blocks, threads_per_block,
                (32 * 32 + 32) * sizeof(half), stream>>>(
                feat_low_ptr, feat_deep_ptr, weight_ptr, bias_ptr,
                output_ptr, height, width);
            break;
        default:
            TORCH_CHECK(false, "FP16: Only 32 and 48 channels supported");
            break;
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// ============================================
// Extended Hand-Optimized Kernels
// 16ch, 28ch, 52ch - Added for specific model requirements
// ============================================

// ==================== 16通道 FP32 版本 ====================
__global__ void span_attention_16ch_optimized_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __shared__ float weight_shared[16 * 16];
    __shared__ float bias_shared[16];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    for (int i = tid; i < 16 * 16; i += 256) {
        weight_shared[i] = weight[i];
    }
    if (tid < 16) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();
    
    float f3[16], x[16];
    
    if (valid) {
        #pragma unroll 8
        for (int c = 0; c < 16; ++c) {
            int idx = c * spatial_stride + spatial_offset;
            f3[c] = feat_deep[idx];
            x[c] = feat_low[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 8
        for (int c_out = 0; c_out < 16; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll 2
            for (int i = 0; i < 2; ++i) {
                int base = i * 8;
                guidance += weight_shared[c_out * 16 + base + 0] * f3[base + 0]
                          + weight_shared[c_out * 16 + base + 1] * f3[base + 1]
                          + weight_shared[c_out * 16 + base + 2] * f3[base + 2]
                          + weight_shared[c_out * 16 + base + 3] * f3[base + 3]
                          + weight_shared[c_out * 16 + base + 4] * f3[base + 4]
                          + weight_shared[c_out * 16 + base + 5] * f3[base + 5]
                          + weight_shared[c_out * 16 + base + 6] * f3[base + 6]
                          + weight_shared[c_out * 16 + base + 7] * f3[base + 7];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x[c_out] + f3[c_out]) * guidance;
        }
    }
}

// ==================== 28通道 FP32 版本 ====================
__global__ void span_attention_28ch_optimized_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __shared__ float weight_shared[28 * 28];
    __shared__ float bias_shared[28];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    constexpr int WEIGHT_SIZE = 28 * 28;
    for (int i = tid; i < WEIGHT_SIZE; i += 256) {
        weight_shared[i] = weight[i];
    }
    if (tid < 28) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();
    
    float f3[28], x[28];
    
    if (valid) {
        #pragma unroll 7
        for (int c = 0; c < 28; ++c) {
            int idx = c * spatial_stride + spatial_offset;
            f3[c] = feat_deep[idx];
            x[c] = feat_low[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 7
        for (int c_out = 0; c_out < 28; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll 7
            for (int i = 0; i < 7; ++i) {
                int base = i * 4;
                guidance += weight_shared[c_out * 28 + base + 0] * f3[base + 0]
                          + weight_shared[c_out * 28 + base + 1] * f3[base + 1]
                          + weight_shared[c_out * 28 + base + 2] * f3[base + 2]
                          + weight_shared[c_out * 28 + base + 3] * f3[base + 3];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x[c_out] + f3[c_out]) * guidance;
        }
    }
}

// ==================== 52通道 FP32 版本 ====================
__global__ void span_attention_52ch_optimized_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __shared__ float weight_shared[52 * 52];
    __shared__ float bias_shared[52];
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * blockDim.x + tid;
    const int h = hw / width;
    const int w = hw % width;
    const bool valid = (h < height && w < width);
    
    const int spatial_stride = height * width;
    const int spatial_offset = valid ? (h * width + w) : 0;
    
    constexpr int WEIGHT_SIZE = 52 * 52;
    for (int i = tid; i < WEIGHT_SIZE; i += 256) {
        weight_shared[i] = weight[i];
    }
    if (tid < 52) {
        bias_shared[tid] = bias[tid];
    }
    __syncthreads();
    
    float f3[52], x[52];
    
    if (valid) {
        #pragma unroll 13
        for (int c = 0; c < 52; ++c) {
            int idx = c * spatial_stride + spatial_offset;
            f3[c] = feat_deep[idx];
            x[c] = feat_low[idx];
        }
    }
    
    if (valid) {
        #pragma unroll 13
        for (int c_out = 0; c_out < 52; ++c_out) {
            float guidance = bias_shared[c_out];
            
            #pragma unroll 13
            for (int i = 0; i < 13; ++i) {
                int base = i * 4;
                guidance += weight_shared[c_out * 52 + base + 0] * f3[base + 0]
                          + weight_shared[c_out * 52 + base + 1] * f3[base + 1]
                          + weight_shared[c_out * 52 + base + 2] * f3[base + 2]
                          + weight_shared[c_out * 52 + base + 3] * f3[base + 3];
            }
            
            output[c_out * spatial_stride + spatial_offset] = (x[c_out] + f3[c_out]) * guidance;
        }
    }
}

// ==================== 扩展接口实现 ====================
at::Tensor span_attention_forward_cuda_16ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias) {
    
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);
    auto output = at::empty_like(feat_low);
    
    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    const int total_pixels = height * width;
    constexpr int threads = 256;
    const int blocks = (total_pixels + threads - 1) / threads;
    
    span_attention_16ch_optimized_kernel<<<blocks, threads,
        (16 * 16 + 16) * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        feat_low.data_ptr<float>(), feat_deep.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), height, width);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor span_attention_forward_cuda_28ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias) {
    
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);
    auto output = at::empty_like(feat_low);
    
    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    const int total_pixels = height * width;
    constexpr int threads = 256;
    const int blocks = (total_pixels + threads - 1) / threads;
    
    span_attention_28ch_optimized_kernel<<<blocks, threads,
        (28 * 28 + 28) * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        feat_low.data_ptr<float>(), feat_deep.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), height, width);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor span_attention_forward_cuda_52ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias) {
    
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);
    auto output = at::empty_like(feat_low);
    
    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    const int total_pixels = height * width;
    constexpr int threads = 256;
    const int blocks = (total_pixels + threads - 1) / threads;
    
    span_attention_52ch_optimized_kernel<<<blocks, threads,
        (52 * 52 + 52) * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        feat_low.data_ptr<float>(), feat_deep.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), height, width);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
