#ifndef SPAN_ATTENTION_KERNEL_CUH
#define SPAN_ATTENTION_KERNEL_CUH

#include <torch/extension.h>

// ============================================
// 原始优化版本 - 支持 32/48 通道, batch_size=1
// ============================================

// FP32 优化版本前向传播
at::Tensor span_attention_forward_cuda_optimized(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// FP16 版本前向传播
at::Tensor span_attention_forward_cuda_half(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// 手工优化扩展版本 - 16, 28, 52 通道
// ============================================

extern "C" {
// Kernel declarations (defined in .cu files)
void span_attention_16ch_optimized_kernel(
    const float* feat_low, const float* feat_deep,
    const float* weight, const float* bias,
    float* output, int height, int width);

void span_attention_28ch_optimized_kernel(
    const float* feat_low, const float* feat_deep,
    const float* weight, const float* bias,
    float* output, int height, int width);

void span_attention_52ch_optimized_kernel(
    const float* feat_low, const float* feat_deep,
    const float* weight, const float* bias,
    float* output, int height, int width);
}

// C++ wrapper functions (defined in .cu files)
at::Tensor span_attention_forward_cuda_16ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias);

at::Tensor span_attention_forward_cuda_28ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias);

at::Tensor span_attention_forward_cuda_52ch(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias);

// ============================================
// 模板优化版本 - 支持 16, 24, 56, 64 通道
// ============================================

// FP32 templated
at::Tensor span_attention_forward_cuda_templated_fp32(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias);

// FP16 templated
at::Tensor span_attention_forward_cuda_templated_fp16(
    at::Tensor feat_low, at::Tensor feat_deep,
    at::Tensor weight, at::Tensor bias);

// ============================================
// 通用版本 - 支持任意通道数和 batch size
// ============================================

// FP32 通用版本
at::Tensor span_attention_forward_cuda_general_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// FP16 通用版本
at::Tensor span_attention_forward_cuda_general_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// 统一接口 - 自动选择优化或通用版本
at::Tensor span_attention_forward_cuda_unified(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// 判断是否应该使用优化路径
bool should_use_optimized_path(int channels, int batch_size);

// ============================================
// V2 高度优化版本 (保留但可能不需要)
// ============================================

// FP32 V2 版本: Block协作 + Warp Shuffle
at::Tensor span_attention_forward_cuda_v2_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// FP16 V2 版本: half2向量加载 + Warp Shuffle
at::Tensor span_attention_forward_cuda_v2_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// V2 自动选择版本
at::Tensor span_attention_forward_cuda_v2(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// V3 高度优化版本 (保留但可能不需要)
// ============================================

// FP32 V3 版本: 优化的共享内存使用和向量化加载
at::Tensor span_attention_forward_cuda_v3_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// Mixed V3 版本: FP16存储 + FP32计算
at::Tensor span_attention_forward_cuda_v3_mixed(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// V3 自动选择版本
at::Tensor span_attention_forward_cuda_v3(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// V4 高度优化版本 - 32通道专用，使用 __ldg 缓存
// ============================================

at::Tensor span_attention_forward_cuda_v4_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

at::Tensor span_attention_forward_cuda_v4_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// Warp-Level 优化版本 - 32通道专用
// ============================================

at::Tensor span_attention_forward_cuda_warp_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// Opt2 优化版本 - 针对 H800 优化
// ============================================

at::Tensor span_attention_forward_cuda_opt2_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

at::Tensor span_attention_forward_cuda_opt2_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// Opt3 极致优化版本 - 32通道专用
// ============================================

at::Tensor span_attention_forward_cuda_opt3_fp32(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

at::Tensor span_attention_forward_cuda_opt3_fp16(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

// ============================================
// 算子融合版本 - 32通道专用
// ============================================

at::Tensor span_attention_fused_forward_cuda(
    at::Tensor feat_low,
    at::Tensor feat_deep,
    at::Tensor weight,
    at::Tensor bias);

#endif
