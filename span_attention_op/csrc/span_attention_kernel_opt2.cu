#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================
// 优化版本 2: 针对 H800 (SM90) 优化
// 主要优化点:
// 1. __ldg() 使用只读缓存加载特征数据
// 2. float4 向量化加载 weight
// 3. 调整 launch bounds 提高 occupancy
// 4. 减少寄存器压力，避免溢出
// ============================================

// ==================== 32通道 FP32 优化版 ====================
__global__ void __launch_bounds__(256, 4) span_attention_32ch_opt2_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    // 共享内存: weight[32*32] + bias[32]
    __align__(16) __shared__ float smem[32 * 32 + 32];
    float* weight_s = smem;
    float* bias_s = smem + 32 * 32;
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * 256 + tid;
    const int h = hw / width;
    const int w = hw % width;
    
    const int stride = height * width;
    
    // 向量化加载 weight 到共享内存 (32*32 = 1024 floats = 256 float4)
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* weight_s_vec = reinterpret_cast<float4*>(weight_s);
    #pragma unroll
    for (int i = tid; i < 256; i += 256) {
        weight_s_vec[i] = weight_vec[i];
    }
    // 加载 bias (32 floats = 8 float4)
    if (tid < 8) {
        reinterpret_cast<float4*>(bias_s)[tid] = reinterpret_cast<const float4*>(bias)[tid];
    }
    __syncthreads();
    
    // 边界检查
    if (h >= height || w >= width) return;
    
    const int offset = h * width + w;
    
    // 使用 __ldg 加载特征数据 (利用 texture cache)
    // 预加载到寄存器
    float f3[32], x[32];
    
    #pragma unroll
    for (int c = 0; c < 32; ++c) {
        x[c] = __ldg(&feat_low[c * stride + offset]);
        f3[c] = __ldg(&feat_deep[c * stride + offset]);
    }
    
    // 计算 guidance 和 output
    #pragma unroll
    for (int c_out = 0; c_out < 32; ++c_out) {
        const float* w_row = weight_s + c_out * 32;
        
        // 矩阵向量乘: guidance = sum(w_row * f3) + bias
        float g = bias_s[c_out];
        
        // 完全展开，减少循环开销
        g += w_row[0] * f3[0] + w_row[1] * f3[1] + w_row[2] * f3[2] + w_row[3] * f3[3];
        g += w_row[4] * f3[4] + w_row[5] * f3[5] + w_row[6] * f3[6] + w_row[7] * f3[7];
        g += w_row[8] * f3[8] + w_row[9] * f3[9] + w_row[10] * f3[10] + w_row[11] * f3[11];
        g += w_row[12] * f3[12] + w_row[13] * f3[13] + w_row[14] * f3[14] + w_row[15] * f3[15];
        g += w_row[16] * f3[16] + w_row[17] * f3[17] + w_row[18] * f3[18] + w_row[19] * f3[19];
        g += w_row[20] * f3[20] + w_row[21] * f3[21] + w_row[22] * f3[22] + w_row[23] * f3[23];
        g += w_row[24] * f3[24] + w_row[25] * f3[25] + w_row[26] * f3[26] + w_row[27] * f3[27];
        g += w_row[28] * f3[28] + w_row[29] * f3[29] + w_row[30] * f3[30] + w_row[31] * f3[31];
        
        output[c_out * stride + offset] = (x[c_out] + f3[c_out]) * g;
    }
}

// ==================== 32通道 FP16 优化版 ====================
__global__ void __launch_bounds__(256, 4) span_attention_32ch_half_opt2_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ half smem[32 * 32 + 32];
    half* weight_s = smem;
    half* bias_s = smem + 32 * 32;
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * 256 + tid;
    const int h = hw / width;
    const int w = hw % width;
    
    const int stride = height * width;
    
    // 使用 half2 向量化加载
    const half2* weight_vec = reinterpret_cast<const half2*>(weight);
    half2* weight_s_vec = reinterpret_cast<half2*>(weight_s);
    #pragma unroll
    for (int i = tid; i < 512; i += 256) {
        weight_s_vec[i] = weight_vec[i];
    }
    if (tid < 16) {
        reinterpret_cast<half2*>(bias_s)[tid] = reinterpret_cast<const half2*>(bias)[tid];
    }
    __syncthreads();
    
    if (h >= height || w >= width) return;
    
    const int offset = h * width + w;
    const half* feat_low_ptr = reinterpret_cast<const half*>(feat_low);
    const half* feat_deep_ptr = reinterpret_cast<const half*>(feat_deep);
    
    half f3[32], x[32];
    
    #pragma unroll
    for (int c = 0; c < 32; ++c) {
        x[c] = __ldg(&feat_low_ptr[c * stride + offset]);
        f3[c] = __ldg(&feat_deep_ptr[c * stride + offset]);
    }
    
    #pragma unroll
    for (int c_out = 0; c_out < 32; ++c_out) {
        const half* w_row = weight_s + c_out * 32;
        
        half g = bias_s[c_out];
        
        // 使用 half2 向量化乘加
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            half2 w = reinterpret_cast<const half2*>(w_row)[i];
            half2 f = reinterpret_cast<half2*>(f3)[i];
            g = __hfma(w.x, f.x, g);
            g = __hfma(w.y, f.y, g);
        }
        
        half sum = __hadd(x[c_out], f3[c_out]);
        output[c_out * stride + offset] = __hmul(sum, g);
    }
}

// ==================== 48通道 FP32 优化版 ====================
__global__ void __launch_bounds__(256, 2) span_attention_48ch_opt2_kernel(
    const float* __restrict__ feat_low,
    const float* __restrict__ feat_deep,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ float smem[48 * 48 + 48];
    float* weight_s = smem;
    float* bias_s = smem + 48 * 48;
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * 256 + tid;
    const int h = hw / width;
    const int w = hw % width;
    
    const int stride = height * width;
    
    // 向量化加载 weight (48*48 = 2304 floats = 576 float4)
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    float4* weight_s_vec = reinterpret_cast<float4*>(weight_s);
    #pragma unroll 4
    for (int i = tid; i < 576; i += 256) {
        weight_s_vec[i] = weight_vec[i];
    }
    if (tid < 12) {
        reinterpret_cast<float4*>(bias_s)[tid] = reinterpret_cast<const float4*>(bias)[tid];
    }
    __syncthreads();
    
    if (h >= height || w >= width) return;
    
    const int offset = h * width + w;
    
    float f3[48], x[48];
    
    #pragma unroll 8
    for (int c = 0; c < 48; ++c) {
        x[c] = __ldg(&feat_low[c * stride + offset]);
        f3[c] = __ldg(&feat_deep[c * stride + offset]);
    }
    
    #pragma unroll 6
    for (int c_out = 0; c_out < 48; ++c_out) {
        const float* w_row = weight_s + c_out * 48;
        
        float g = bias_s[c_out];
        
        #pragma unroll 6
        for (int i = 0; i < 6; ++i) {
            int base = i * 8;
            g += w_row[base + 0] * f3[base + 0] + w_row[base + 1] * f3[base + 1];
            g += w_row[base + 2] * f3[base + 2] + w_row[base + 3] * f3[base + 3];
            g += w_row[base + 4] * f3[base + 4] + w_row[base + 5] * f3[base + 5];
            g += w_row[base + 6] * f3[base + 6] + w_row[base + 7] * f3[base + 7];
        }
        
        output[c_out * stride + offset] = (x[c_out] + f3[c_out]) * g;
    }
}

// ==================== 48通道 FP16 优化版 ====================
__global__ void __launch_bounds__(256, 2) span_attention_48ch_half_opt2_kernel(
    const at::Half* __restrict__ feat_low,
    const at::Half* __restrict__ feat_deep,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
    int height, int width) {
    
    __align__(16) __shared__ half smem[48 * 48 + 48];
    half* weight_s = smem;
    half* bias_s = smem + 48 * 48;
    
    const int tid = threadIdx.x;
    const int hw = blockIdx.x * 256 + tid;
    const int h = hw / width;
    const int w = hw % width;
    
    const int stride = height * width;
    
    const half2* weight_vec = reinterpret_cast<const half2*>(weight);
    half2* weight_s_vec = reinterpret_cast<half2*>(weight_s);
    #pragma unroll 4
    for (int i = tid; i < 1152; i += 256) {
        weight_s_vec[i] = weight_vec[i];
    }
    if (tid < 24) {
        reinterpret_cast<half2*>(bias_s)[tid] = reinterpret_cast<const half2*>(bias)[tid];
    }
    __syncthreads();
    
    if (h >= height || w >= width) return;
    
    const int offset = h * width + w;
    const half* feat_low_ptr = reinterpret_cast<const half*>(feat_low);
    const half* feat_deep_ptr = reinterpret_cast<const half*>(feat_deep);
    
    half f3[48], x[48];
    
    #pragma unroll 8
    for (int c = 0; c < 48; ++c) {
        x[c] = __ldg(&feat_low_ptr[c * stride + offset]);
        f3[c] = __ldg(&feat_deep_ptr[c * stride + offset]);
    }
    
    #pragma unroll 6
    for (int c_out = 0; c_out < 48; ++c_out) {
        const half* w_row = weight_s + c_out * 48;
        
        half g = bias_s[c_out];
        
        #pragma unroll 6
        for (int i = 0; i < 24; ++i) {
            half2 w = reinterpret_cast<const half2*>(w_row)[i];
            half2 f = reinterpret_cast<half2*>(f3)[i];
            g = __hfma(w.x, f.x, g);
            g = __hfma(w.y, f.y, g);
        }
        
        half sum = __hadd(x[c_out], f3[c_out]);
        output[c_out * stride + offset] = __hmul(sum, g);
    }
}

// ==================== PyTorch 接口 ====================
at::Tensor span_attention_forward_cuda_opt2_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D");
    TORCH_CHECK(feat_low.size(0) == 1, "Only batch_size=1 supported");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Float, "FP32 required");

    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);
    if (height == 0 || width == 0) return output;

    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int total = height * width;
    const int blocks = (total + 255) / 256;

    if (channels == 32) {
        span_attention_32ch_opt2_kernel<<<blocks, 256,
            (32 * 32 + 32) * sizeof(float), stream>>>(
            feat_low.data_ptr<float>(), feat_deep.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            output.data_ptr<float>(), height, width);
    } else if (channels == 48) {
        span_attention_48ch_opt2_kernel<<<blocks, 256,
            (48 * 48 + 48) * sizeof(float), stream>>>(
            feat_low.data_ptr<float>(), feat_deep.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            output.data_ptr<float>(), height, width);
    } else {
        TORCH_CHECK(false, "Only 32 and 48 channels supported");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor span_attention_forward_cuda_opt2_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias) {

    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D");
    TORCH_CHECK(feat_low.size(0) == 1, "Only batch_size=1 supported");
    TORCH_CHECK(feat_low.scalar_type() == at::ScalarType::Half, "FP16 required");

    const int channels = feat_low.size(1);
    const int height = feat_low.size(2);
    const int width = feat_low.size(3);

    auto output = at::empty_like(feat_low);
    if (height == 0 || width == 0) return output;

    feat_low = feat_low.contiguous();
    feat_deep = feat_deep.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int total = height * width;
    const int blocks = (total + 255) / 256;

    if (channels == 32) {
        span_attention_32ch_half_opt2_kernel<<<blocks, 256,
            (32 * 32 + 32) * sizeof(half), stream>>>(
            feat_low.data_ptr<at::Half>(), feat_deep.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(), bias.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(), height, width);
    } else if (channels == 48) {
        span_attention_48ch_half_opt2_kernel<<<blocks, 256,
            (48 * 48 + 48) * sizeof(half), stream>>>(
            feat_low.data_ptr<at::Half>(), feat_deep.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(), bias.data_ptr<at::Half>(),
            output.data_ptr<at::Half>(), height, width);
    } else {
        TORCH_CHECK(false, "Only 32 and 48 channels supported");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
