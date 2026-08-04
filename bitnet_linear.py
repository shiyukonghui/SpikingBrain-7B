# -*- coding: utf-8 -*-
"""
BitNet 思想 × SpikingBrain 脉冲化 融合模块。

核心思想（来自 BitNet b1.58）：
  1. 三元权重 {-1, 0, +1}：W_ternary = round(W/alpha).clamp(-1,1)，alpha=mean(|W|) 逐通道
     -> matmul 变成稀疏加法（仅非零三元权重参与）
  2. 激活量化：BitNet 用 absmax (s=127/max|x|)，我们融合脉冲化 dynamic_spikes
  3. 输出 y = (x_spiked @ W_ternary) * alpha
"""
import torch
import torch.nn as nn

from W8ASpike.quant_linear import dynamic_spikes


class BitNetSpikeLinear(nn.Module):
    """三元权重 + 脉冲激活 线性层。"""

    def __init__(self, in_features, out_features, bias=False, k=3.0,
                 ternary_threshold=0.0, use_spike=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.ternary_threshold = ternary_threshold
        self.use_spike = use_spike

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        # 逐通道 scale (alpha)
        self.register_buffer("alpha", torch.ones(out_features))
        # 三元权重缓存 {-1,0,+1}
        self.register_buffer("w_ternary", torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quantize_weights(self):
        """BitNet 三元量化：alpha=mean(|W|)，W_ternary=round(W/alpha).clamp(-1,1)。"""
        w = self.weight.detach()
        alpha = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        self.alpha.copy_(alpha.squeeze(-1))
        w_ternary = (w / alpha).round().clamp(-1, 1)
        if self.ternary_threshold > 0:
            # 可选：额外阈值稀疏化三元权重
            w_ternary = torch.where(w.abs() < self.ternary_threshold, 0, w_ternary)
        self.w_ternary.copy_(w_ternary)

    def forward(self, x):
        assert not self.training
        if self.use_spike:
            spikes_int, vth = dynamic_spikes(x, self.k)
            x = (spikes_int * vth).to(x.dtype)
        # 稀疏加法：x @ W_ternary，再乘 alpha
        out = torch.nn.functional.linear(x, self.w_ternary, None)
        out = out * self.alpha.unsqueeze(0)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_to_bitnet_spike(model, k=3.0, use_spike=True, ternary_threshold=0.0,
                            spike_attn=True, spike_mlp=False):
    """把 nn.Linear 替换为 BitNetSpikeLinear（三元权重 + 可选脉冲激活）。"""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            is_attn = "self_attn" in name
            is_mlp = "mlp" in name
            spike_this = (spike_attn and is_attn) or (spike_mlp and is_mlp)
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = BitNetSpikeLinear(module.in_features, module.out_features,
                                    bias=module.bias is not None, k=k,
                                    ternary_threshold=ternary_threshold,
                                    use_spike=spike_this)
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            new.quantize_weights()
            setattr(parent, attr, new)
            count += 1
    model.eval()
    return count
