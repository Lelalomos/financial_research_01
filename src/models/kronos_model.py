"""
Kronos model components integrated into the local src.models package.

This keeps the original Kronos model code available from the project package
without copying tests or example scripts into the main runtime path.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from src.config import load_config

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # pragma: no cover
    class PyTorchModelHubMixin:  # type: ignore[no-redef]
        """Fallback mixin when huggingface_hub is not installed."""

        pass

from .kronos_module import (
    BSQuantizer,
    DependencyAwareLayer,
    DualHead,
    HierarchicalEmbedding,
    RMSNorm,
    TemporalEmbedding,
    TransformerBlock,
)


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_in,
        d_model,
        n_heads,
        ff_dim,
        n_enc_layers,
        n_dec_layers,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        s1_bits,
        s2_bits,
        beta,
        gamma0,
        gamma,
        zeta,
        group_size,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.enc_layers - 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.dec_layers - 1)
            ]
        )
        self.quant_embed = nn.Linear(self.d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(self.s1_bits, self.d_model)
        self.post_quant_embed = nn.Linear(self.codebook_dim, self.d_model)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x):
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, : self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        if half:
            x1 = x[0]
            x2 = x[1]
            mask1 = 2 ** torch.arange(self.s1_bits, device=x1.device, dtype=torch.long)
            mask2 = 2 ** torch.arange(self.s2_bits, device=x2.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask1) != 0
            x2 = (x2.unsqueeze(-1) & mask2) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0

        x = x.float() * 2 - 1
        q_scale = 1.0 / (self.codebook_dim ** 0.5)
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        _, _, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        s1_bits,
        s2_bits,
        n_layers,
        d_model,
        n_heads,
        ff_dim,
        ffn_dropout_p,
        attn_dropout_p,
        resid_dropout_p,
        token_dropout_p,
        learn_te,
        num_stocks=1,
        num_groups=1,
        use_stock_embedding=False,
        use_group_embedding=False,
        stock_emb_dim=0,
        group_emb_dim=0,
    ):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p
        self.use_stock_embedding = use_stock_embedding
        self.use_group_embedding = use_group_embedding

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        if self.use_stock_embedding:
            self.stock_embedding = nn.Embedding(max(int(num_stocks), 1), stock_emb_dim)
            self.stock_projection = nn.Linear(stock_emb_dim, self.d_model)
        else:
            self.stock_embedding = None
            self.stock_projection = None
        if self.use_group_embedding:
            self.group_embedding = nn.Embedding(max(int(num_groups), 1), group_emb_dim)
            self.group_projection = nn.Linear(group_emb_dim, self.d_model)
        else:
            self.group_embedding = None
            self.group_projection = None
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.ff_dim,
                    self.ffn_dropout_p,
                    self.attn_dropout_p,
                    self.resid_dropout_p,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _build_stamp_from_day_month(self, day, month):
        if day is None or month is None:
            return None
        stamp = torch.zeros(day.shape[0], day.shape[1], 5, device=day.device, dtype=day.dtype)
        stamp[:, :, 3] = day
        stamp[:, :, 4] = month
        return stamp

    def _add_context_embeddings(self, x, stamp=None, stock_id=None, group_id=None, day=None, month=None):
        if stamp is None:
            stamp = self._build_stamp_from_day_month(day, month)
        if stamp is not None:
            x = x + self.time_emb(stamp)
        if self.use_stock_embedding and stock_id is not None:
            x = x + self.stock_projection(self.stock_embedding(stock_id.long()))
        if self.use_group_embedding and group_id is not None:
            x = x + self.group_projection(self.group_embedding(group_id.long()))
        return x

    def forward(
        self,
        s1_ids,
        s2_ids,
        stamp=None,
        padding_mask=None,
        use_teacher_forcing=False,
        s1_targets=None,
        stock_id=None,
        group_id=None,
        day=None,
        month=None,
        dividend_flag=None,
    ):
        x = self.embedding([s1_ids, s2_ids])
        x = self._add_context_embeddings(
            x,
            stamp=stamp,
            stock_id=stock_id,
            group_id=group_id,
            day=day,
            month=month,
        )
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(
        self,
        s1_ids,
        s2_ids,
        stamp=None,
        padding_mask=None,
        stock_id=None,
        group_id=None,
        day=None,
        month=None,
    ):
        x = self.embedding([s1_ids, s2_ids])
        x = self._add_context_embeddings(
            x,
            stamp=stamp,
            stock_id=stock_id,
            group_id=group_id,
            day=day,
            month=month,
        )
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)
        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=5,
    verbose=False,
    stock_id=None,
    group_id=None,
    day=None,
    month=None,
    future_day=None,
    future_month=None,
):
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = (
            x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        )
        y_stamp = (
            y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
        )

        def _repeat_optional_context(tensor):
            if tensor is None:
                return None
            return tensor.unsqueeze(1).repeat(1, sample_count, 1).reshape(-1, tensor.size(1)).to(device)

        stock_id = _repeat_optional_context(stock_id)
        group_id = _repeat_optional_context(group_id)
        day = _repeat_optional_context(day)
        month = _repeat_optional_context(month)
        future_day = _repeat_optional_context(future_day)
        future_month = _repeat_optional_context(future_month)

        x_token = tokenizer.encode(x, half=True)

        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        full_stock = None
        full_group = None
        full_day = None
        full_month = None

        if stock_id is not None:
            future_stock = stock_id[:, -1:].repeat(1, pred_len)
            full_stock = torch.cat([stock_id, future_stock], dim=1)
        if group_id is not None:
            future_group = group_id[:, -1:].repeat(1, pred_len)
            full_group = torch.cat([group_id, future_group], dim=1)
        if day is not None:
            if future_day is None:
                future_day = day[:, -1:].repeat(1, pred_len)
            full_day = torch.cat([day, future_day], dim=1)
        if month is not None:
            if future_month is None:
                future_month = month[:, -1:].repeat(1, pred_len)
            full_month = torch.cat([month, future_month], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx : start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx : start_idx + buffer_len]

        ran = trange if verbose else range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [
                    pre_buffer[:, :window_len],
                    post_buffer[:, :window_len],
                ]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
            current_stock = full_stock[:, context_start:context_end].contiguous() if full_stock is not None else None
            current_group = full_group[:, context_start:context_end].contiguous() if full_group is not None else None
            current_day = full_day[:, context_start:context_end].contiguous() if full_day is not None else None
            current_month = full_month[:, context_start:context_end].contiguous() if full_month is not None else None

            s1_logits, context = model.decode_s1(
                input_tokens[0],
                input_tokens[1],
                current_stamp,
                stock_id=current_stock,
                group_id=current_group,
                day=current_day,
                month=current_month,
            )
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_context = context[:, -1:, :]
            s2_logits = model.decode_s2(s2_context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        if pred_len > max_context:
            raise ValueError(f"pred_len={pred_len} exceeds max_context={max_context} for final decode.")

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def _coerce_timestamps(timestamps: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex) -> pd.Series:
    if isinstance(timestamps, pd.Series):
        series = timestamps
    elif isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = pd.Series(timestamps)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series)
    return series.reset_index(drop=True)


def calc_time_stamps(x_timestamp):
    timestamps = _coerce_timestamps(x_timestamp)
    time_df = pd.DataFrame()
    time_df["minute"] = timestamps.dt.minute
    time_df["hour"] = timestamps.dt.hour
    time_df["weekday"] = timestamps.dt.weekday
    time_df["day"] = timestamps.dt.day
    time_df["month"] = timestamps.dt.month
    return time_df


class KronosPredictor:
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ["open", "high", "low", "close"]
        self.vol_col = "volume"
        self.amt_vol = "amount"
        self.time_cols = ["minute", "hour", "weekday", "day", "month"]

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(
            self.tokenizer,
            self.model,
            x_tensor,
            x_stamp_tensor,
            y_stamp_tensor,
            self.max_context,
            pred_len,
            self.clip,
            T,
            top_k,
            top_p,
            sample_count,
            verbose,
        )
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        x_timestamp = _coerce_timestamps(x_timestamp)
        y_timestamp = _coerce_timestamps(y_timestamp)
        if len(df) != len(x_timestamp):
            raise ValueError(f"Historical data length {len(df)} does not match x_timestamp length {len(x_timestamp)}.")
        if len(y_timestamp) != pred_len:
            raise ValueError(f"y_timestamp length {len(y_timestamp)} does not match pred_len {pred_len}.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(
            y_timestamp_list, (list, tuple)
        ):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_series_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = _coerce_timestamps(x_timestamp_list[i])
            y_timestamp = _coerce_timestamps(y_timestamp_list[i])
            if len(df) != len(x_timestamp):
                raise ValueError(f"Historical data length at index {i} does not match x_timestamp length.")

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            y_stamp_series_list.append(y_timestamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(
                preds_i,
                columns=self.price_cols + [self.vol_col, self.amt_vol],
                index=y_stamp_series_list[i],
            )
            pred_dfs.append(pred_df)

        return pred_dfs


def create_kronos_tokenizer(config=None) -> KronosTokenizer:
    """
    Create a KronosTokenizer from config/model.json.
    """
    if config is None:
        config = load_config("model")

    tokenizer_cfg = config.model.models.kronos.tokenizer
    return KronosTokenizer(
        d_in=tokenizer_cfg.D_IN,
        d_model=tokenizer_cfg.D_MODEL,
        n_heads=tokenizer_cfg.N_HEADS,
        ff_dim=tokenizer_cfg.FF_DIM,
        n_enc_layers=tokenizer_cfg.N_ENC_LAYERS,
        n_dec_layers=tokenizer_cfg.N_DEC_LAYERS,
        ffn_dropout_p=tokenizer_cfg.FFN_DROPOUT_P,
        attn_dropout_p=tokenizer_cfg.ATTN_DROPOUT_P,
        resid_dropout_p=tokenizer_cfg.RESID_DROPOUT_P,
        s1_bits=tokenizer_cfg.S1_BITS,
        s2_bits=tokenizer_cfg.S2_BITS,
        beta=tokenizer_cfg.BETA,
        gamma0=tokenizer_cfg.GAMMA0,
        gamma=tokenizer_cfg.GAMMA,
        zeta=tokenizer_cfg.ZETA,
        group_size=tokenizer_cfg.GROUP_SIZE,
    )


def create_kronos_model(config=None, num_stocks=None, num_groups=None) -> Kronos:
    """
    Create a Kronos model from config/model.json.
    """
    if config is None:
        config = load_config("model")

    network_cfg = config.model.models.kronos.network
    use_stock_embedding = getattr(network_cfg, "USE_STOCK_EMBEDDING", False)
    use_group_embedding = getattr(network_cfg, "USE_GROUP_EMBEDDING", False)

    if num_stocks is None:
        num_stocks = getattr(network_cfg, "NUM_STOCKS", None)
    if num_groups is None:
        num_groups = getattr(network_cfg, "NUM_GROUPS", None)

    if use_stock_embedding and num_stocks is None:
        raise ValueError("Kronos stock embeddings require runtime num_stocks from dataset metadata.")
    if use_group_embedding and num_groups is None:
        raise ValueError("Kronos group embeddings require runtime num_groups from dataset metadata.")

    return Kronos(
        s1_bits=network_cfg.S1_BITS,
        s2_bits=network_cfg.S2_BITS,
        n_layers=network_cfg.N_LAYERS,
        d_model=network_cfg.D_MODEL,
        n_heads=network_cfg.N_HEADS,
        ff_dim=network_cfg.FF_DIM,
        ffn_dropout_p=network_cfg.FFN_DROPOUT_P,
        attn_dropout_p=network_cfg.ATTN_DROPOUT_P,
        resid_dropout_p=network_cfg.RESID_DROPOUT_P,
        token_dropout_p=network_cfg.TOKEN_DROPOUT_P,
        learn_te=network_cfg.LEARN_TE,
        num_stocks=1 if num_stocks is None else int(num_stocks),
        num_groups=1 if num_groups is None else int(num_groups),
        use_stock_embedding=use_stock_embedding,
        use_group_embedding=use_group_embedding,
        stock_emb_dim=getattr(network_cfg, "STOCK_EMB_DIM", 0),
        group_emb_dim=getattr(network_cfg, "GROUP_EMB_DIM", 0),
    )


def create_kronos_predictor(
    model=None,
    tokenizer=None,
    config=None,
    device=None,
    num_stocks=None,
    num_groups=None,
) -> KronosPredictor:
    """
    Create a KronosPredictor from config/model.json.
    """
    if config is None:
        config = load_config("model")

    if tokenizer is None:
        tokenizer = create_kronos_tokenizer(config=config)
    if model is None:
        model = create_kronos_model(config=config, num_stocks=num_stocks, num_groups=num_groups)

    predictor_cfg = config.model.models.kronos.predictor
    return KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=predictor_cfg.MAX_CONTEXT,
        clip=predictor_cfg.CLIP,
    )


__all__ = [
    "KronosTokenizer",
    "Kronos",
    "KronosPredictor",
    "create_kronos_tokenizer",
    "create_kronos_model",
    "create_kronos_predictor",
]
