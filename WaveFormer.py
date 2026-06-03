import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math
from typing import Optional

from base_model import SequenceModel
from wavelet_gpu import GpuWaveletDenoiser


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1
            )
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)), dim=-1
            )
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        # Some market-derived features can still contain NaN after preprocessing
        # (e.g., whole windows missing). Guard here to avoid poisoning softmax.
        gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=0.0, neginf=0.0)
        output = self.trans(gate_input)
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)                                   # (N, T, D)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)            # (N, T)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)              # (N, D)


class WaveFormer(nn.Module):
    """
    WaveFormer core neural network.

    Architecture (in order):
      1. [optional] GpuWaveletDenoiser  — denoises factor features on GPU
      2. Feature Gate                   — market-info guided feature weighting
      3. Linear projection (d_feat → d_model)
      4. Positional Encoding
      5. TAttention                     — intra-stock temporal attention
      6. SAttention                     — inter-stock cross-sectional attention
      7. TemporalAttention              — temporal aggregation
      8. Linear (d_model → 1)
    """

    def __init__(
        self,
        d_feat: int,
        d_model: int,
        output_dim: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        gate_input_start_index: int,
        gate_input_end_index: int,
        beta: float,
        # wavelet denoising
        use_wavelet_denoise: bool = False,
        wavelet: str = "haar",
        denoise_level: Optional[int] = 1,
        threshold_method: str = "bayes",
        threshold_mode: str = "soft",
        threshold_scale: float = 0.3,
        denoise_blend: float = 0.25,
        adaptive_blend: bool = False,
        adaptive_target_ratio: float = 0.08,
        adaptive_sharpness: float = 12.0,
        adaptive_blend_min: float = 0.03,
        adaptive_blend_max: float = 0.55,
        denoise_finest_only: bool = True,
        level_dependent_scale: bool = True,
        use_edge_pad: bool = True,
        use_boundary_smooth: bool = False,
        boundary_smooth_win: int = 1,
        wavelet_collect_stats: bool = True,
    ):
        super(WaveFormer, self).__init__()

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # Optional GPU-native wavelet denoising layer (first op in forward)
        self.wavelet_denoiser: Optional[GpuWaveletDenoiser] = None
        if use_wavelet_denoise:
            self.wavelet_denoiser = GpuWaveletDenoiser(
                wavelet=wavelet,
                level=denoise_level,
                threshold_method=threshold_method,
                threshold_mode=threshold_mode,
                threshold_scale=threshold_scale,
                denoise_blend=denoise_blend,
                adaptive_blend=adaptive_blend,
                adaptive_target_ratio=adaptive_target_ratio,
                adaptive_sharpness=adaptive_sharpness,
                adaptive_blend_min=adaptive_blend_min,
                adaptive_blend_max=adaptive_blend_max,
                denoise_finest_only=denoise_finest_only,
                level_dependent_scale=level_dependent_scale,
                use_edge_pad=use_edge_pad,
                use_boundary_smooth=use_boundary_smooth,
                boundary_smooth_win=boundary_smooth_win,
                feature_start=0,
                feature_end=gate_input_start_index,  # only factor features, not market info
                collect_stats=wavelet_collect_stats,
            )

        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, F_total)  where F_total = d_feat + d_gate_input
        src = x[:, :, :self.gate_input_start_index]         # (N, T, d_feat)
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]

        # GPU wavelet denoising on factor features (stays on the same device)
        if self.wavelet_denoiser is not None:
            src = self.wavelet_denoiser(src)

        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        out = self.layers(src)
        # Keep backward compatibility for single-head setting.
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out

    def consume_runtime_stats(self) -> Optional[dict]:
        if self.wavelet_denoiser is None:
            return None
        return self.wavelet_denoiser.consume_stats(reset=True)


class WaveFormerModel(SequenceModel):
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        gate_input_start_index: int,
        gate_input_end_index: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        beta: float,
        num_label_heads: int = 1,
        head_loss_weights=None,
        primary_head_index: int = 0,
        # wavelet denoising (GPU-native, inside the model)
        use_wavelet_denoise: bool = False,
        wavelet: str = "haar",
        denoise_level: Optional[int] = 1,
        threshold_method: str = "bayes",
        threshold_mode: str = "soft",
        threshold_scale: float = 0.3,
        denoise_blend: float = 0.25,
        adaptive_blend: bool = False,
        adaptive_target_ratio: float = 0.08,
        adaptive_sharpness: float = 12.0,
        adaptive_blend_min: float = 0.03,
        adaptive_blend_max: float = 0.55,
        denoise_finest_only: bool = True,
        level_dependent_scale: bool = True,
        use_edge_pad: bool = True,
        use_boundary_smooth: bool = False,
        boundary_smooth_win: int = 1,
        wavelet_collect_stats: bool = True,
        **kwargs,
    ):
        super(WaveFormerModel, self).__init__(
            num_label_heads=num_label_heads,
            head_loss_weights=head_loss_weights,
            primary_head_index=primary_head_index,
            **kwargs,
        )
        self.d_model = d_model
        self.d_feat = d_feat
        self.num_label_heads = num_label_heads
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.use_wavelet_denoise = use_wavelet_denoise
        self.wavelet = wavelet
        self.denoise_level = denoise_level
        self.threshold_method = threshold_method
        self.threshold_mode = threshold_mode
        self.threshold_scale = threshold_scale
        self.denoise_blend = denoise_blend
        self.adaptive_blend = adaptive_blend
        self.adaptive_target_ratio = adaptive_target_ratio
        self.adaptive_sharpness = adaptive_sharpness
        self.adaptive_blend_min = adaptive_blend_min
        self.adaptive_blend_max = adaptive_blend_max
        self.denoise_finest_only = denoise_finest_only
        self.level_dependent_scale = level_dependent_scale
        self.use_edge_pad = use_edge_pad
        self.use_boundary_smooth = use_boundary_smooth
        self.boundary_smooth_win = boundary_smooth_win
        self.wavelet_collect_stats = wavelet_collect_stats

        self.init_model()

    def init_model(self):
        self.model = WaveFormer(
            d_feat=self.d_feat,
            d_model=self.d_model,
            output_dim=self.num_label_heads,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            use_wavelet_denoise=self.use_wavelet_denoise,
            wavelet=self.wavelet,
            denoise_level=self.denoise_level,
            threshold_method=self.threshold_method,
            threshold_mode=self.threshold_mode,
            threshold_scale=self.threshold_scale,
            denoise_blend=self.denoise_blend,
            adaptive_blend=self.adaptive_blend,
            adaptive_target_ratio=self.adaptive_target_ratio,
            adaptive_sharpness=self.adaptive_sharpness,
            adaptive_blend_min=self.adaptive_blend_min,
            adaptive_blend_max=self.adaptive_blend_max,
            denoise_finest_only=self.denoise_finest_only,
            level_dependent_scale=self.level_dependent_scale,
            use_edge_pad=self.use_edge_pad,
            use_boundary_smooth=self.use_boundary_smooth,
            boundary_smooth_win=self.boundary_smooth_win,
            wavelet_collect_stats=self.wavelet_collect_stats,
        )
        super(WaveFormerModel, self).init_model()
