#!/usr/bin/env python
"""
Plot real_target and predict_target from the latest Excel report in outputs/.

Selection rules:
1. Prefer the newest Excel file whose name contains "final" or "best".
2. If none exist, fall back to the newest Excel file in the output directory.
3. Find the first worksheet containing both "real_target" and "predict_target".
4. Group rows by ticker when available, otherwise by stock_id.
5. Save one PNG per stock group in the output directory.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PRIORITY_TOKENS = ("final", "best")
REQUIRED_COLUMNS = ["real_target", "predict_target"]
GROUP_COLUMNS = ("ticker", "stock_id")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot real_target and predict_target from the latest Excel output report."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing Excel reports and where the image will be saved.",
    )
    parser.add_argument(
        "--excel-file",
        type=Path,
        default=None,
        help="Optional explicit Excel file path. When omitted, the script auto-selects one.",
    )
    parser.add_argument(
        "--image-file",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to outputs/<excel_stem>_plot.png.",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Optional chunk size for splitting each stock plot by row index, e.g. 100 -> rows 0-99, 100-199.",
    )
    return parser.parse_args()


def list_excel_files(output_dir: Path):
    return [
        path for path in output_dir.glob("*.xlsx")
        if path.is_file() and not path.name.startswith(".~lock")
    ]


def select_latest_excel(output_dir: Path) -> Path:
    excel_files = list_excel_files(output_dir)
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {output_dir}")

    tagged_files = [
        path for path in excel_files
        if any(token in path.stem.lower() for token in PRIORITY_TOKENS)
    ]

    candidates = tagged_files if tagged_files else excel_files
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_plot_dataframe(excel_path: Path):
    workbook = pd.ExcelFile(excel_path)

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if all(column in df.columns for column in REQUIRED_COLUMNS):
            selected_columns = list(REQUIRED_COLUMNS)
            for group_column in GROUP_COLUMNS:
                if group_column in df.columns:
                    selected_columns.insert(0, group_column)
                    break
            plot_df = df.loc[:, selected_columns].dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
            if plot_df.empty:
                raise ValueError(
                    f"Sheet '{sheet_name}' in {excel_path} has the required columns but no usable rows."
                )
            return plot_df, sheet_name

    raise ValueError(
        f"No worksheet in {excel_path} contains both {REQUIRED_COLUMNS[0]} and {REQUIRED_COLUMNS[1]}."
    )


def build_default_image_path(excel_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{excel_path.stem}_plot.png"


def sanitize_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value.strip())
    return cleaned or "unknown"


def resolve_group_column(plot_df: pd.DataFrame):
    for group_column in GROUP_COLUMNS:
        if group_column in plot_df.columns:
            return group_column
    return None


def create_plot(plot_df: pd.DataFrame, excel_path: Path, sheet_name: str, image_path: Path, group_label: str):
    fig, ax = plt.subplots(figsize=(14, 7))
    x_axis = range(len(plot_df))

    ax.plot(x_axis, plot_df["real_target"], label="real_target", linewidth=1.2)
    ax.plot(x_axis, plot_df["predict_target"], label="predict_target", linewidth=1.2)

    ax.set_title(f"real_target vs predict_target\n{group_label} | {excel_path.name} [{sheet_name}]")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Target Value")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=200)
    plt.close(fig)


def chunk_dataframe(plot_df: pd.DataFrame, chunk_size: int):
    if chunk_size is None:
        yield 0, len(plot_df), plot_df
        return

    if chunk_size <= 0:
        raise ValueError("--split-index must be a positive integer.")

    for start in range(0, len(plot_df), chunk_size):
        end = min(start + chunk_size, len(plot_df))
        yield start, end, plot_df.iloc[start:end].reset_index(drop=True)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()

    excel_path = args.excel_file.resolve() if args.excel_file else select_latest_excel(output_dir)
    plot_df, sheet_name = load_plot_dataframe(excel_path)
    group_column = resolve_group_column(plot_df)

    saved_images = []
    if group_column is None:
        for start, end, chunk_df in chunk_dataframe(plot_df, args.split_index):
            if args.image_file and args.split_index is None:
                image_path = args.image_file.resolve()
            else:
                if args.split_index is None:
                    image_path = build_default_image_path(excel_path, output_dir)
                else:
                    image_path = output_dir / f"{excel_path.stem}_plot_{start}_{end - 1}.png"
            create_plot(chunk_df, excel_path, sheet_name, image_path, f"all_rows [{start}-{end - 1}]")
            saved_images.append(image_path)
    else:
        image_dir = output_dir / f"{excel_path.stem}_plots"
        for group_value, group_df in plot_df.groupby(group_column, sort=True):
            group_name = sanitize_filename(str(group_value))
            group_plot_df = group_df.loc[:, REQUIRED_COLUMNS].reset_index(drop=True)
            for start, end, chunk_df in chunk_dataframe(group_plot_df, args.split_index):
                if args.split_index is None:
                    image_path = image_dir / f"{group_name}_plot.png"
                else:
                    image_path = image_dir / f"{group_name}_plot_{start}_{end - 1}.png"
                create_plot(
                    chunk_df,
                    excel_path,
                    sheet_name,
                    image_path,
                    f"{group_column}={group_value} [{start}-{end - 1}]",
                )
                saved_images.append(image_path)

    print(f"Selected Excel file: {excel_path}")
    print(f"Selected sheet: {sheet_name}")
    if args.split_index is not None:
        print(f"Split index: {args.split_index}")
    if group_column is None:
        print(f"Saved images: {len(saved_images)}")
        print(f"Image output: {saved_images[0].parent if len(saved_images) > 1 else saved_images[0]}")
    else:
        print(f"Grouped by: {group_column}")
        print(f"Saved images: {len(saved_images)}")
        print(f"Image directory: {saved_images[0].parent}")


if __name__ == "__main__":
    main()
