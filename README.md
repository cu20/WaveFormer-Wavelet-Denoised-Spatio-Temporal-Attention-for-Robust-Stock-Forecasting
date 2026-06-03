# WaveFormer

**Wavelet-Denoised Spatio-Temporal Attention for Robust Stock Forecasting**

基于小波去噪与时空注意力机制的 A 股量化选股模型。本仓库提供 **WaveFormer 核心实现**（PyTorch），并通过 [qlib](https://github.com/cu20/qlib) 子仓库完成 **Alpha158 特征、训练、信号分析与组合回测** 的完整工作流。

---

## 主要特性

| 模块 | 说明 |
|------|------|
| **时空注意力** | 时间维 `TAttention` + 截面维 `SAttention` + `TemporalAttention`，对因子序列建模 |
| **市场门控** | 利用 Alpha158 中市场状态特征（索引 158–221）对因子通道做门控加权 |
| **GPU 小波去噪** | `GpuWaveletDenoiser`：Haar 提升方案、Bayes/Visu 阈值、软/硬/半软阈值、残差混合，全程在 GPU 上运行 |
| **归一化前去噪** | `wavelet_processor.WaveletDenoiseProcessor` 作为 Qlib `infer_processors` 的一环（pre-norm） |
| **消融实验** | `main_waveformer.py` 支持 2-way / 3-way 对比：无去噪、归一化前去噪、归一化后去噪 |
| **Qlib 集成** | `WaveFormerQlibModel` 对接 `workflow` 记录 IC / 回测收益等指标 |

---

## 仓库结构

```
WaveFormer/                          # 本仓库（模型核心）
├── WaveFormer.py                    # 网络结构：WaveFormer + WaveFormerModel
├── base_model.py                    # 训练循环、评估、保存
├── wavelet_gpu.py                   # GPU 小波去噪层
├── wavelet_denoise.py               # CPU 小波去噪（独立脚本用）
├── wavelet_processor.py             # Qlib 数据处理器（pre-norm）
├── main.py                          # 使用本地 pkl 数据集的简易训练入口
├── check_qlib_data_range.py         # 检查 qlib 本地数据日期范围
├── 命令行说明.md                     # Qlib 工作流命令详解（中文）
├── requirements.txt
└── qlib/                            # 独立 Git 仓库（未纳入本仓库 git 追踪）
    └── examples/benchmarks/Waveformer/
        ├── main_waveformer.py       # 推荐：训练 + 回测 + 消融
        ├── workflow_config_waveformer_Alpha158.yaml
        └── ...
```

> **说明**：根目录 `.gitignore` 忽略整个 `qlib/` 目录；修改 benchmark 或 `waveformer_ts.py` 请在 `qlib` 目录内单独 `git commit` 并推送到 [cu20/qlib](https://github.com/cu20/qlib)。

---

## 环境要求

- Python 3.8+
- PyTorch（建议 CUDA）
- [Microsoft Qlib](https://github.com/microsoft/qlib)（工作流路径）
- 其他依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
pip install pyqlib  # 或通过 qlib 子仓库源码安装
```

---

## 数据准备

### Qlib 中国 A 股数据（推荐）

1. 按 [Qlib 官方文档](https://github.com/microsoft/qlib#data-preparation) 下载 `cn_data` 到本地，默认路径：

   ```text
   ~/.qlib/qlib_data/cn_data
   ```

2. 核对数据末日是否与配置一致：

   ```bash
   cd /path/to/WaveFormer
   python check_qlib_data_range.py
   python check_qlib_data_range.py --provider_uri ~/.qlib/qlib_data/cn_data --market csi300
   ```

3. 若更新了数据，请同步修改 `qlib/examples/benchmarks/Waveformer/workflow_config_waveformer_Alpha158.yaml` 中的 `end_time`、测试段与 `backtest.end_time`，并删除过期的 `handler_*.pkl` 缓存后重新训练。

### 独立 `main.py` 路径

需自行准备 `data/opensource/{csi300|csi800}_dl_{train|valid|test}.pkl`（见 `main.py` 内路径约定）。

---

## 快速开始（Qlib 工作流）

工作目录：

```bash
cd /path/to/WaveFormer/qlib/examples/benchmarks/Waveformer
```

### 单次训练 + 回测

使用默认配置（可在 YAML 中设置 `use_pre_norm_wavelet` / `use_wavelet_denoise`）：

```bash
python main_waveformer.py
```

指定配置与日志目录：

```bash
python main_waveformer.py \
  --config ./workflow_config_waveformer_Alpha158.yaml \
  --log_dir logs
```

若 `model/` 下已有与当前 `save_prefix`、分组、seed 一致的 checkpoint，将**跳过训练**并直接加载做信号与回测。

**Checkpoint 命名示例**（`save_prefix` 默认为 `csi300`）：

| 配置 | 权重文件 |
|------|----------|
| `use_pre_norm_wavelet: true` | `model/csi300_pre_norm_{seed}.pkl` |
| 无 pre-norm，无模型内去噪 | `model/csi300_baseline_{seed}.pkl` |
| 仅模型内 after-norm 去噪 | `model/csi300_after_norm_{seed}.pkl` |

> 同时开启 pre-norm 与模型内去噪时，脚本会只使用 pre-norm 数据并关闭模型内去噪，避免双重去噪。

### 消融实验

```bash
# 2-way：无去噪 vs 归一化后去噪
python main_waveformer.py --ablation --ablation_mode 2way --seeds 0

# 3-way：无去噪 vs 归一化前去噪 vs 归一化后去噪
python main_waveformer.py --ablation --ablation_mode 3way --seeds 0 1 2

# 仅回测（需已有对应权重）
python main_waveformer.py --only_backtest --ablation --ablation_mode 3way --seeds 0
```

结果默认写入 `logs/ablation_results.txt`。

### 小波去噪命令行参数（节选）

| 参数 | 含义 | 默认 |
|------|------|------|
| `--wavelet` | 小波基 | `haar` |
| `--denoise_level` | 分解层数 | `1` |
| `--threshold_method` | `bayes` / `visu` | `bayes` |
| `--threshold_mode` | `soft` / `hard` / `semisoft` | `soft` |
| `--threshold_scale` | 阈值缩放 | `0.3` |
| `--denoise_blend` | 残差混合系数 | `0.25` |
| `--use_edge_pad` / `--no_edge_pad` | 边界填充 | 开启 |
| `--use_boundary_smooth` | 边界平滑 | 关闭 |

更完整的参数表、数据划分与回测区间说明见 **[命令行说明.md](./命令行说明.md)**。

---

## 快速开始（独立脚本）

不依赖 Qlib workflow、使用本地 pickle 数据集时：

```bash
cd /path/to/WaveFormer
python main.py
```

在 `main.py` 中可设置 `universe`（`csi300` / `csi800`）、`feature_denoiser` 与 `WaveFormerModel` 超参。

---

## 模型结构概览

```
输入 x: (N, T, F_total)
  ├─ 因子特征 src = x[:, :, :158]
  ├─ [可选] GpuWaveletDenoiser(src)     # 归一化后（模型内）去噪
  ├─ 市场门控 × feature_gate(market)   # market = x[:, -1, 158:221]
  └─ Linear → PosEncoding → TAttention → SAttention → TemporalAttention → 预测
```

- **CSI300** 默认 `beta=5`，**CSI800** 默认 `beta=2`（门控温度）。
- Pre-norm 路径在 **RobustZScoreNorm 之前** 对原始特征做小波去噪（CPU，`wavelet_processor`）。

---

## 开发与实验产物

以下目录/文件通常不应提交 Git（已在 `.gitignore` 中配置）：

- `mlruns/`、`logs/`、`model/*.pkl`、`handler_*.pkl`
- 大规模 `*.csv` / 回测导出、benchmark 下自动生成的 `*.png`

图表排版工具（本地生成六宫格等）：

```bash
python layout_six_panel_2x3.py   # 在 Waveformer 示例目录下
```

---

## 相关链接

| 资源 | 地址 |
|------|------|
| 本仓库 | https://github.com/cu20/WaveFormer-Wavelet-Denoised-Spatio-Temporal-Attention-for-Robust-Stock-Forecasting |
| Qlib 定制 fork | https://github.com/cu20/qlib |
| Qlib 上游 | https://github.com/microsoft/qlib |

---

## 引用

若本工作对您有帮助，请引用本仓库（并说明使用了 Qlib 数据与评测框架）。正式论文信息可在发表后补充至此处。

```bibtex
@misc{waveformer2025,
  title  = {WaveFormer: Wavelet-Denoised Spatio-Temporal Attention for Robust Stock Forecasting},
  author = {cu20},
  year   = {2025},
  url    = {https://github.com/cu20/WaveFormer-Wavelet-Denoised-Spatio-Temporal-Attention-for-Robust-Stock-Forecasting}
}
```

---

## 常见问题

1. **修改了 pre-norm 或 handler 配置但结果不变**  
   删除对应的 `handler_*_pre_norm.pkl`（或相关 `handler_*.pkl`）后重新运行，否则会读旧缓存。

2. **OpenMP 冲突**  
   `main.py` 已设置 `KMP_DUPLICATE_LIB_OK=TRUE`；若仍报错，请检查 MKL/OpenMP 环境变量。

3. **qlib 与主仓库如何协作**  
   模型定义在根目录 `WaveFormer.py`；Qlib 封装在 `qlib/qlib/contrib/model/waveformer_ts.py`。开发时确保 `main_waveformer.py` 已将项目根目录加入 `sys.path`（脚本内已配置）。

---

## 许可证

请参阅仓库发布时的 License 文件；未包含 License 前，默认保留所有权利，使用前请联系作者。
