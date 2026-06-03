import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader, Sampler
import torch
import torch.optim as optim


def calc_ic(pred, label):
    df = pd.DataFrame({"pred": pred, "label": label})
    ic = df["pred"].corr(df["label"])
    ric = df["pred"].corr(df["label"], method="spearman")
    return ic, ric


def zscore(x):
    return (x - x.mean()).div(x.std())


def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]


class DailyBatchSamplerRandom(Sampler):
    """Groups samples by date so each batch contains all stocks for one day."""

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.daily_count = (
            pd.Series(index=self.data_source.get_index())
            .groupby("datetime")
            .size()
            .values
        )
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel:
    def __init__(
        self,
        n_epochs: int,
        lr: float,
        GPU=None,
        seed=None,
        train_stop_loss_thred=None,
        save_path: str = "model/",
        save_prefix: str = "",
        num_label_heads: int = 1,
        head_loss_weights=None,
        primary_head_index: int = 0,
        log_valid_ic: bool = True,
        use_valid_for_ckpt_selection: bool = True,
        use_master_label_process: bool = False,
        use_master_valid_loss: bool = False,
        save_master_suffix: bool = False,
    ):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

        self.fitted = -1
        self.model = None
        self.train_optimizer = None
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.num_label_heads = int(num_label_heads)
        if self.num_label_heads <= 0:
            raise ValueError("num_label_heads must be >= 1")
        self.primary_head_index = int(primary_head_index)
        if not (0 <= self.primary_head_index < self.num_label_heads):
            raise ValueError("primary_head_index out of range")
        if head_loss_weights is None:
            self.head_loss_weights = [1.0 / self.num_label_heads] * self.num_label_heads
        else:
            if len(head_loss_weights) != self.num_label_heads:
                raise ValueError("head_loss_weights length must equal num_label_heads")
            weights = np.asarray(head_loss_weights, dtype=float)
            if np.any(weights < 0):
                raise ValueError("head_loss_weights must be non-negative")
            if float(weights.sum()) <= 0:
                raise ValueError("sum(head_loss_weights) must be > 0")
            self.head_loss_weights = (weights / weights.sum()).tolist()
        self.log_valid_ic = bool(log_valid_ic)
        self.use_valid_for_ckpt_selection = bool(use_valid_for_ckpt_selection)
        self.use_master_label_process = bool(use_master_label_process)
        self.use_master_valid_loss = bool(use_master_valid_loss)
        self.save_master_suffix = bool(save_master_suffix)

    def init_model(self):
        if self.model is None:
            raise ValueError("模型未初始化")
        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask] - label[mask]) ** 2
        return torch.mean(loss)

    def _split_feature_and_labels(self, data):
        """
        Split batched tensor into feature tensor and label tensor.

        This method is made robust to config/data mismatch:
        - configured `num_label_heads` may be 2 while dataset still has 1 label col
        - or vice versa
        In that case we fallback to the actual number of label columns inferred
        from model feature boundary (`gate_input_end_index`) when available.
        """
        total_dim = int(data.shape[2])
        feature_dim_hint = getattr(self, "gate_input_end_index", None)
        if isinstance(feature_dim_hint, int) and feature_dim_hint > 0 and total_dim > feature_dim_hint:
            inferred_label_heads = total_dim - feature_dim_hint
            actual_heads = max(1, min(self.num_label_heads, inferred_label_heads))
            feature = data[:, :, :feature_dim_hint]
            labels = data[:, -1, feature_dim_hint : feature_dim_hint + actual_heads]
            return feature, labels, actual_heads

        # Generic fallback for models without explicit feature boundary.
        actual_heads = max(1, min(self.num_label_heads, total_dim - 1))
        feature = data[:, :, 0 : total_dim - actual_heads]
        labels = data[:, -1, total_dim - actual_heads : total_dim]
        return feature, labels, actual_heads

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            # data: (N, T, F)  where F = features + 1 label column
            feature, labels, actual_heads = self._split_feature_and_labels(data)
            primary_idx = min(self.primary_head_index, actual_heads - 1)
            label_main = labels[:, primary_idx]

            if self.use_master_label_process:
                # MASTER-compatible: no drop_extreme / no extra label zscore in training loop.
                feature = feature.to(self.device)
                labels = labels.to(self.device)
            else:
                # Remove extreme label values and z-score normalise
                mask, label_main = drop_extreme(label_main)
                feature = feature[mask, :, :]
                labels = labels[mask, :]
                labels = torch.stack([zscore(labels[:, i]) for i in range(actual_heads)], dim=1)

                feature = feature.to(self.device)
                labels = labels.to(self.device)

            pred = self.model(feature.float())
            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)
            loss = 0.0
            for i in range(actual_heads):
                w = self.head_loss_weights[i] if i < len(self.head_loss_weights) else 1.0 / float(actual_heads)
                loss = loss + w * self.loss_fn(pred[:, i], labels[:, i])
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature, labels, actual_heads = self._split_feature_and_labels(data)
            primary_idx = min(self.primary_head_index, actual_heads - 1)
            feature = feature.to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float())
                if pred.ndim == 2:
                    pred = pred[:, primary_idx]
            if self.use_master_label_process:
                label = labels[:, primary_idx].to(self.device)
                loss = self.loss_fn(pred, label)
            else:
                label = labels[:, primary_idx]
                mask, label = drop_na(label)
                label = zscore(label)
                label = label.to(self.device)
                mask = mask.to(self.device)
                loss = self.loss_fn(pred[mask], label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # pin_memory=True enables DMA async transfer, reducing CPU-GPU latency
        use_pin = self.device.type == "cuda"
        data_loader = DataLoader(
            data,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=use_pin,
            num_workers=0,
        )
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 1

    def fit(self, dl_train, dl_valid=None):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        best_param = None
        best_epoch = -1
        best_valid_ic = -np.inf
        best_val_loss = np.inf
        valid_loader = None
        if dl_valid is not None and self.use_master_valid_loss:
            valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            self.fitted = step
            extra_log = ""
            if hasattr(self.model, "consume_runtime_stats"):
                try:
                    stats = self.model.consume_runtime_stats()
                except Exception:
                    stats = None
                if stats:
                    raw_abs = float(stats.get("raw_abs_mean", 0.0))
                    delta_abs = float(stats.get("delta_abs_mean", 0.0))
                    energy_ratio = float(stats.get("delta_energy_ratio", 0.0))
                    eff_blend = float(stats.get("effective_blend_mean", 0.0))
                    noise_ratio = float(stats.get("noise_ratio_mean", 0.0))
                    rel = (delta_abs / (raw_abs + 1e-12)) if raw_abs > 0 else 0.0
                    extra_log = (
                        " | denoise_delta_abs %.6f (%.2f%% of raw), denoise_delta_energy %.4f, eff_blend %.4f, noise_ratio %.4f"
                        % (delta_abs, rel * 100.0, energy_ratio, eff_blend, noise_ratio)
                    )

            if dl_valid:
                if self.use_master_valid_loss:
                    val_loss = self.test_epoch(valid_loader)
                    if self.log_valid_ic:
                        _pred, metrics = self.predict(dl_valid)
                        print(
                            "Epoch %d, train_loss %.6f, valid_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f.%s"
                            % (
                                step,
                                train_loss,
                                val_loss,
                                metrics["IC"],
                                metrics["ICIR"],
                                metrics["RIC"],
                                metrics["RICIR"],
                                extra_log,
                            )
                        )
                    else:
                        print("Epoch %d, train_loss %.6f, valid_loss %.6f%s" % (step, train_loss, val_loss, extra_log))
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = step
                        best_param = copy.deepcopy(self.model.state_dict())
                else:
                    predictions, metrics = self.predict(dl_valid)
                    if self.log_valid_ic:
                        print(
                            "Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f.%s"
                            % (
                                step,
                                train_loss,
                                metrics["IC"],
                                metrics["ICIR"],
                                metrics["RIC"],
                                metrics["RICIR"],
                                extra_log,
                            )
                        )
                    if self.use_valid_for_ckpt_selection:
                        # Keep checkpoint with highest validation IC.
                        if pd.notna(metrics["IC"]) and metrics["IC"] > best_valid_ic:
                            best_valid_ic = metrics["IC"]
                            best_epoch = step
                            best_param = copy.deepcopy(self.model.state_dict())
                    else:
                        # Monitoring only: keep latest checkpoint, no valid-based selection.
                        best_epoch = step
                        best_param = copy.deepcopy(self.model.state_dict())
            else:
                print("Epoch %d, train_loss %.6f%s" % (step, train_loss, extra_log))
                # MASTER-aligned train-only behavior:
                # do not refresh checkpoint each epoch; only checkpoint when threshold is met.

            if self.train_stop_loss_thred is not None and train_loss <= self.train_stop_loss_thred:
                # Optional early stop by loss threshold; keep current best checkpoint.
                if best_param is None:
                    best_param = copy.deepcopy(self.model.state_dict())
                    best_epoch = step
                break

        if best_param is None:
            best_param = copy.deepcopy(self.model.state_dict())
            best_epoch = self.fitted

        import os
        os.makedirs(self.save_path, exist_ok=True)
        if self.save_master_suffix:
            torch.save(best_param, f"{self.save_path}{self.save_prefix}master_{self.seed}.pkl")
        else:
            torch.save(best_param, f"{self.save_path}/{self.save_prefix}_{self.seed}.pkl")

        # Load best checkpoint for subsequent prediction/backtest in current run.
        self.model.load_state_dict(best_param)
        self.fitted = best_epoch

    def predict(self, dl_test):
        if self.fitted < 0:
            raise ValueError("模型还未训练!")
        else:
            print("Epoch:", self.fitted)

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature, labels, actual_heads = self._split_feature_and_labels(data)
            primary_idx = min(self.primary_head_index, actual_heads - 1)
            label = labels[:, primary_idx]

            feature = feature.to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())
                if pred.ndim == 2:
                    pred = pred[:, primary_idx]
                pred = pred.detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        metrics = {
            "IC": np.mean(ic),
            "ICIR": np.mean(ic) / np.std(ic),
            "RIC": np.mean(ric),
            "RICIR": np.mean(ric) / np.std(ric),
        }

        # Print IC summary automatically
        print(
            f"  IC={metrics['IC']:.4f}  ICIR={metrics['ICIR']:.3f}"
            f"  RIC={metrics['RIC']:.4f}  RICIR={metrics['RICIR']:.3f}"
        )
        return predictions, metrics
