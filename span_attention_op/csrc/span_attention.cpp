#include <torch/extension.h>
#include "span_attention_kernel.cuh"

// ==================================================
// 主接口：支持任意通道数和 batch size
// 自动在以下实现间选择：
//   - 高度优化的手工 kernel (16, 28, 32, 48, 52ch, bs=1)
//   - 模板优化的 kernel (24, 56, 64ch, bs=1)
//   - 通用 kernel (任意配置)
// ==================================================
torch::Tensor span_attention(
    torch::Tensor feat_low,
    torch::Tensor feat_deep,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Check inputs
    TORCH_CHECK(feat_low.is_cuda(), "feat_low must be a CUDA tensor");
    TORCH_CHECK(feat_deep.is_cuda(), "feat_deep must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_deep.dim() == 4, "feat_deep must be 4D [N, C, H, W]");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D [C, C, 1, 1]");
    TORCH_CHECK(feat_low.sizes() == feat_deep.sizes(), 
                "feat_low and feat_deep must have same shape");
    TORCH_CHECK(weight.size(0) == weight.size(1), 
                "Weight must be square matrix [C, C, 1, 1]");
    TORCH_CHECK(weight.size(0) == feat_low.size(1), 
                "Weight channels must match input channels");
    TORCH_CHECK(weight.size(0) == bias.size(0), 
                "Bias must match weight output channels");

    // 提取参数
    const int batch_size = feat_low.size(0);
    const int channels = feat_low.size(1);
    
    // 检查数据类型
    bool is_half = (feat_low.dtype() == torch::kHalf);
    
    // 如果输入是FP16，确保weight和bias也是FP16
    if (is_half) {
        weight = weight.to(feat_low.dtype());
        bias = bias.to(feat_low.dtype());
    }
    
    // 调整 weight 和 bias 的维度 (在类型转换之后)
    auto weight_2d = weight.squeeze(-1).squeeze(-1);  // [C, C, 1, 1] -> [C, C]
    auto bias_1d = bias.view({-1});  // [C, 1, 1, 1] or [C] -> [C]
    
    // 高度优化的手工 kernel: 16, 28, 32, 48, 52ch (bs=1)
    if (batch_size == 1) {
        switch (channels) {
            case 16:
                return span_attention_forward_cuda_16ch(
                    feat_low, feat_deep, weight_2d, bias_1d);
            case 28:
                return span_attention_forward_cuda_28ch(
                    feat_low, feat_deep, weight_2d, bias_1d);
            case 32:
                if (is_half) {
                    return span_attention_forward_cuda_opt2_fp16(
                        feat_low, feat_deep, weight_2d, bias_1d);
                } else {
                    return span_attention_forward_cuda_opt2_fp32(
                        feat_low, feat_deep, weight_2d, bias_1d);
                }
            case 48:
                if (is_half) {
                    return span_attention_forward_cuda_opt2_fp16(
                        feat_low, feat_deep, weight_2d, bias_1d);
                } else {
                    return span_attention_forward_cuda_opt2_fp32(
                        feat_low, feat_deep, weight_2d, bias_1d);
                }
            case 52:
                return span_attention_forward_cuda_52ch(
                    feat_low, feat_deep, weight_2d, bias_1d);
            default:
                break;  // Fall through to templated/general
        }
    }
    
    // 模板优化的 kernel: 24, 56, 64ch (bs=1)
    bool use_templated = (channels == 24 || channels == 56 || channels == 64) 
                         && batch_size == 1;
    
    if (use_templated) {
        if (is_half) {
            return span_attention_forward_cuda_templated_fp16(
                feat_low, feat_deep, weight_2d, bias_1d);
        } else {
            return span_attention_forward_cuda_templated_fp32(
                feat_low, feat_deep, weight_2d, bias_1d);
        }
    }
    
    // 通用 kernel (任意配置)
    if (is_half) {
        return span_attention_forward_cuda_general_fp16(
            feat_low, feat_deep, weight_2d, bias_1d);
    } else {
        return span_attention_forward_cuda_general_fp32(
            feat_low, feat_deep, weight_2d, bias_1d);
    }
}

// ==================================================
// 显式接口：允许用户强制选择实现
// ==================================================

// 优化版本接口
// 支持: 16, 28, 32, 48, 52ch (bs=1)
torch::Tensor span_attention_optimized(
    torch::Tensor feat_low,
    torch::Tensor feat_deep,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    TORCH_CHECK(feat_low.is_cuda(), "feat_low must be a CUDA tensor");
    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    TORCH_CHECK(feat_low.size(0) == 1, "Optimized version requires batch_size=1");
    
    int channels = feat_low.size(1);
    auto weight_2d = weight.squeeze(-1).squeeze(-1);
    auto bias_1d = bias.view({-1});
    
    switch (channels) {
        case 16:
            return span_attention_forward_cuda_16ch(feat_low, feat_deep, weight_2d, bias_1d);
        case 28:
            return span_attention_forward_cuda_28ch(feat_low, feat_deep, weight_2d, bias_1d);
        case 32:
        case 48:
            if (feat_low.dtype() == torch::kHalf) {
                return span_attention_forward_cuda_half(feat_low, feat_deep, weight_2d, bias_1d);
            } else {
                return span_attention_forward_cuda_optimized(feat_low, feat_deep, weight_2d, bias_1d);
            }
        case 52:
            return span_attention_forward_cuda_52ch(feat_low, feat_deep, weight_2d, bias_1d);
        default:
            TORCH_CHECK(false, "Optimized version supports 16, 28, 32, 48, 52 channels only");
    }
}

// 通用版本接口 (任意配置)
torch::Tensor span_attention_general(
    torch::Tensor feat_low,
    torch::Tensor feat_deep,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    TORCH_CHECK(feat_low.is_cuda(), "feat_low must be a CUDA tensor");
    TORCH_CHECK(feat_low.dim() == 4, "feat_low must be 4D [N, C, H, W]");
    
    auto weight_2d = weight.squeeze(-1).squeeze(-1);
    auto bias_1d = bias.view({-1});
    
    if (feat_low.dtype() == torch::kHalf) {
        return span_attention_forward_cuda_general_fp16(feat_low, feat_deep, weight_2d, bias_1d);
    } else {
        return span_attention_forward_cuda_general_fp32(feat_low, feat_deep, weight_2d, bias_1d);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("span_attention", &span_attention, 
          "Span attention: unified interface with hand-optimized kernels for 16/28/32/48/52ch.\n"
          "Automatically selects best implementation.\n"
          "Args:\n"
          "    feat_low: shallow features [N, C, H, W]\n"
          "    feat_deep: deep features [N, C, H, W]\n"
          "    weight: guidance weights [C, C, 1, 1]\n"
          "    bias: guidance bias [C, 1, 1, 1]\n"
          "Returns:\n"
          "    output: attention output [N, C, H, W]");
    
    m.def("span_attention_optimized", &span_attention_optimized,
          "Span attention: hand-optimized for 16, 28, 32, 48, 52 channels.");
    
    m.def("span_attention_general", &span_attention_general,
          "Span attention: general version for any configuration.");
}
