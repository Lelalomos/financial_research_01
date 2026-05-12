#!/usr/bin/env python
"""
Dataset correlation analysis for the training dataset.

Generates:
- feature/target correlation CSV files
- highly correlated pair CSV files
- correlation heatmaps and bar charts
- a Markdown summary report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset correlations")
    parser.add_argument(
        "--pre-data",
        default="data/pre_normalized.parquet",
        help="Pre-normalized parquet dataset",
    )
    parser.add_argument(
        "--normalized-data",
        default="data/normalized_data.parquet",
        help="Normalized parquet dataset",
    )
    parser.add_argument(
        "--info",
        default="data/processed/info.json",
        help="Processed dataset metadata JSON",
    )
    parser.add_argument(
        "--main-config",
        default="config/main.json",
        help="Main config JSON for split ratios",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/correlation_analysis",
        help="Directory for saved plots and CSV reports",
    )
    parser.add_argument(
        "--pair-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for high-correlation feature pairs",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top target-correlated features to chart",
    )
    return parser.parse_args()


def classify_feature(name: str) -> str:
    if name in {"close", "high", "low", "open", "volume"}:
        return "price"
    if name.startswith("ema_"):
        return "ema"
    if name.startswith("rsi"):
        return "rsi"
    if name.startswith("stochrsi"):
        return "stochrsi"
    if name.startswith("macd"):
        return "macd"
    if name.startswith("CDL"):
        return "candlestick"
    if name in {
        "pe_ratio",
        "peg_ratio",
        "eps",
        "roe",
        "roi",
        "debt_to_equity",
        "debt_to_asset",
        "current_ratio",
    }:
        return "financial"
    if name in {"vix", "Gold", "Copper", "Corn", "Soybeans", "Cocoa", "Silver", "bondyield"}:
        return "macro"
    return "other"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_train_mask(pre_df: pd.DataFrame, train_ratio: float) -> pd.Series:
    unique_dates = np.sort(pre_df["date"].dropna().unique())
    train_end = int(len(unique_dates) * train_ratio)
    train_dates = set(unique_dates[:train_end])
    return pre_df["date"].isin(train_dates)


def make_target_corr_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    corr = df[feature_cols + ["target"]].corr(numeric_only=True)["target"].drop("target")
    out = corr.to_frame(name="target_correlation")
    out["abs_target_correlation"] = out["target_correlation"].abs()
    out["feature_group"] = [classify_feature(name) for name in out.index]
    return out.sort_values("abs_target_correlation", ascending=False)


def make_high_corr_pairs(df: pd.DataFrame, feature_cols: list[str], threshold: float) -> pd.DataFrame:
    corr = df[feature_cols].corr(numeric_only=True).abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_correlation"})
        .sort_values("abs_correlation", ascending=False)
    )
    return pairs[pairs["abs_correlation"] >= threshold].reset_index(drop=True)


def save_bar_chart(table: pd.DataFrame, title: str, output_path: Path, top_n: int) -> None:
    top = table.head(top_n).iloc[::-1]
    colors = ["#2f6db3" if v >= 0 else "#b33f46" for v in top["target_correlation"]]

    plt.figure(figsize=(12, 9))
    plt.barh(top.index, top["target_correlation"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Pearson correlation with target")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_heatmap(corr_df: pd.DataFrame, title: str, output_path: Path, figsize: tuple[int, int]) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_group_chart(table: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    group_summary = (
        table.groupby("feature_group")["abs_target_correlation"]
        .agg(["count", "mean", "median", "max"])
        .sort_values("mean", ascending=False)
    )
    plt.figure(figsize=(10, 6))
    plt.bar(group_summary.index, group_summary["mean"], color="#2f6db3")
    plt.title("Mean absolute target correlation by feature group")
    plt.ylabel("Mean |correlation|")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return group_summary


def save_sparse_feature_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stats = []
    for col in feature_cols:
        series = df[col]
        stats.append(
            {
                "feature": col,
                "n_unique": int(series.nunique(dropna=True)),
                "zero_ratio": float((series == 0).mean()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else np.nan,
            }
        )
    out = pd.DataFrame(stats).sort_values(["n_unique", "std"], ascending=[True, True])
    return out


def write_summary(
    output_path: Path,
    feature_cols: list[str],
    pre_train_rows: int,
    normalized_train_rows: int,
    n_dates: int,
    n_tickers: int,
    pre_target: pd.Series,
    norm_target: pd.Series,
    pre_target_corr: pd.DataFrame,
    norm_target_corr: pd.DataFrame,
    pre_pairs: pd.DataFrame,
    norm_pairs: pd.DataFrame,
    group_summary: pd.DataFrame,
    sparse_features: pd.DataFrame,
    threshold: float,
) -> None:
    def format_corr_block(table: pd.DataFrame, n: int = 10) -> str:
        return "\n".join(
            f"- `{idx}`: {row['target_correlation']:.4f}"
            for idx, row in table.head(n).iterrows()
        )

    def format_negative_block(table: pd.DataFrame, n: int = 10) -> str:
        negative = table.sort_values("target_correlation").head(n)
        return "\n".join(
            f"- `{idx}`: {row['target_correlation']:.4f}"
            for idx, row in negative.iterrows()
        )

    top_sparse = sparse_features.head(15)
    sparse_block = "\n".join(
        f"- `{row.feature}`: unique={row.n_unique}, zero_ratio={row.zero_ratio:.4f}, std={row.std:.6f}"
        for row in top_sparse.itertuples()
    )

    pre_pair_block = "\n".join(
        f"- `{row.feature_a}` vs `{row.feature_b}`: {row.abs_correlation:.4f}"
        for row in pre_pairs.head(15).itertuples()
    ) or "- None"
    norm_pair_block = "\n".join(
        f"- `{row.feature_a}` vs `{row.feature_b}`: {row.abs_correlation:.4f}"
        for row in norm_pairs.head(15).itertuples()
    ) or "- None"
    group_block = "\n".join(
        f"- `{idx}`: mean={row['mean']:.4f}, median={row['median']:.4f}, max={row['max']:.4f}, count={int(row['count'])}"
        for idx, row in group_summary.iterrows()
    )

    summary = f"""# Correlation Analysis Summary

## Dataset Scope

- Features analyzed: {len(feature_cols)}
- Train rows (pre-normalized): {pre_train_rows}
- Train rows (normalized): {normalized_train_rows}
- Unique train dates: {n_dates}
- Unique tickers: {n_tickers}
- High-correlation threshold: {threshold}

## Target Distribution

- Pre-normalized target mean/std: {pre_target.mean():.6f} / {pre_target.std():.6f}
- Pre-normalized target min/max: {pre_target.min():.6f} / {pre_target.max():.6f}
- Normalized target mean/std: {norm_target.mean():.6f} / {norm_target.std():.6f}
- Normalized target min/max: {norm_target.min():.6f} / {norm_target.max():.6f}

## Strongest Positive Correlations With Target (Pre-Normalized Train)

{format_corr_block(pre_target_corr)}

## Strongest Negative Correlations With Target (Pre-Normalized Train)

{format_negative_block(pre_target_corr)}

## Strongest Positive Correlations With Target (Normalized Train)

{format_corr_block(norm_target_corr)}

## Strongest Negative Correlations With Target (Normalized Train)

{format_negative_block(norm_target_corr)}

## Feature Group Signal Strength

{group_block}

## Highly Correlated Feature Pairs (Pre-Normalized Train)

{pre_pair_block}

## Highly Correlated Feature Pairs (Normalized Train)

{norm_pair_block}

## Sparse Or Low-Variance Features

{sparse_block}

## Interpretation Notes

- Correlation is linear and univariate. Low Pearson correlation does not mean a feature is useless to a nonlinear sequence model.
- Highly correlated price and moving-average features indicate strong redundancy. These features can still help sequence models, but they raise multicollinearity risk for simpler models and interpretability workflows.
- Sparse candlestick features often have near-zero mean correlation individually. That is common because many patterns trigger rarely.
- Normalized and pre-normalized target correlation rankings should be compared together. Large rank shifts can indicate the transform is changing feature scale relationships materially.
"""
    output_path.write_text(summary, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = load_json(Path(args.info))
    main_config = load_json(Path(args.main_config))
    feature_cols = info["feature_cols"]
    train_ratio = main_config["data"]["splits"]["TRAIN_RATIO"]

    pre_cols = ["date", "tic", "target"] + feature_cols
    norm_cols = feature_cols + ["target"]

    pre_df = pd.read_parquet(args.pre_data, columns=pre_cols)
    norm_df = pd.read_parquet(args.normalized_data, columns=norm_cols)

    if len(pre_df) != len(norm_df):
        raise ValueError("Pre-normalized and normalized datasets have different row counts")

    train_mask = build_train_mask(pre_df, train_ratio)
    pre_train = pre_df.loc[train_mask].copy()
    norm_train = norm_df.loc[train_mask.to_numpy()].copy()

    pre_target_corr = make_target_corr_table(pre_train, feature_cols)
    norm_target_corr = make_target_corr_table(norm_train, feature_cols)
    pre_pairs = make_high_corr_pairs(pre_train, feature_cols, args.pair_threshold)
    norm_pairs = make_high_corr_pairs(norm_train, feature_cols, args.pair_threshold)
    group_summary = save_group_chart(
        pre_target_corr,
        output_dir / "pre_train_feature_group_abs_target_correlation.png",
    )
    sparse_features = save_sparse_feature_table(pre_train, feature_cols)

    pre_target_corr.to_csv(output_dir / "pre_train_feature_target_correlation.csv")
    norm_target_corr.to_csv(output_dir / "normalized_train_feature_target_correlation.csv")
    pre_pairs.to_csv(output_dir / "pre_train_high_correlation_pairs.csv", index=False)
    norm_pairs.to_csv(output_dir / "normalized_train_high_correlation_pairs.csv", index=False)
    group_summary.to_csv(output_dir / "pre_train_feature_group_summary.csv")
    sparse_features.to_csv(output_dir / "pre_train_feature_sparsity_summary.csv", index=False)

    save_bar_chart(
        pre_target_corr,
        "Top absolute target correlations (pre-normalized train)",
        output_dir / "pre_train_top_target_correlations.png",
        args.top_n,
    )
    save_bar_chart(
        norm_target_corr,
        "Top absolute target correlations (normalized train)",
        output_dir / "normalized_train_top_target_correlations.png",
        args.top_n,
    )

    pre_corr_all = pre_train[feature_cols + ["target"]].corr(numeric_only=True)
    top40 = pre_target_corr.head(min(40, len(pre_target_corr))).index.tolist()
    save_heatmap(
        pre_corr_all.loc[feature_cols + ["target"], feature_cols + ["target"]],
        "Pre-normalized train correlation heatmap (all features + target)",
        output_dir / "pre_train_full_correlation_heatmap.png",
        figsize=(28, 24),
    )
    save_heatmap(
        pre_corr_all.loc[top40 + ["target"], top40 + ["target"]],
        "Pre-normalized train correlation heatmap (top 40 target-related features)",
        output_dir / "pre_train_top40_correlation_heatmap.png",
        figsize=(18, 16),
    )

    if not pre_pairs.empty:
        pair_features = sorted(set(pre_pairs.head(30)["feature_a"]).union(set(pre_pairs.head(30)["feature_b"])))
        save_heatmap(
            pre_corr_all.loc[pair_features, pair_features],
            "Pre-normalized train high-correlation feature cluster",
            output_dir / "pre_train_high_correlation_cluster.png",
            figsize=(16, 14),
        )

    write_summary(
        output_dir / "correlation_summary.md",
        feature_cols=feature_cols,
        pre_train_rows=len(pre_train),
        normalized_train_rows=len(norm_train),
        n_dates=pre_train["date"].nunique(),
        n_tickers=pre_train["tic"].nunique(),
        pre_target=pre_train["target"],
        norm_target=norm_train["target"],
        pre_target_corr=pre_target_corr,
        norm_target_corr=norm_target_corr,
        pre_pairs=pre_pairs,
        norm_pairs=norm_pairs,
        group_summary=group_summary,
        sparse_features=sparse_features,
        threshold=args.pair_threshold,
    )

    print(f"Saved correlation analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
