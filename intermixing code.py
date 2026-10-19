#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats


RADII = (2, 4, 8, 16, 32)
THRESHOLD_MULTIPLIERS = (0.8, 1.0, 1.2)
GREEN = "#238b45"
RED = "#cb3a32"
INK = "#262626"
GREY = "#8a8a8a"


def otsu_uint8(array: np.ndarray) -> int:
    hist = np.bincount(array.ravel(), minlength=256).astype(float)
    probability = hist / hist.sum()
    weight = np.cumsum(probability)
    moment = np.cumsum(probability * np.arange(256))
    total_moment = moment[-1]
    variance = (total_moment * weight - moment) ** 2 / np.maximum(
        weight * (1.0 - weight), 1e-30
    )
    variance[(weight <= 0) | (weight >= 1)] = -1
    return int(np.argmax(variance))


def load_stack(channel_dir: Path, rgb_index: int) -> np.ndarray:
    files = sorted(channel_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No TIFFs found under {channel_dir}")
    return np.stack(
        [np.asarray(Image.open(path), dtype=np.uint8)[..., rgb_index] for path in files]
    )


def segment_exclusive(
    wt: np.ndarray, cheater: np.ndarray, threshold_multiplier: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Segment both channels and resolve optical overlap to one label per voxel."""
    wt_otsu = otsu_uint8(wt)
    cheater_otsu = otsu_uint8(cheater)
    wt_threshold = wt_otsu * threshold_multiplier
    cheater_threshold = cheater_otsu * threshold_multiplier
    wt_mask = wt > wt_threshold
    cheater_mask = cheater > cheater_threshold

    overlap = wt_mask & cheater_mask
    overlap_fraction = overlap.sum() / max((wt_mask | cheater_mask).sum(), 1)
    if overlap.any():
        wt_hi = np.percentile(wt[wt > wt_otsu], 99.5)
        cheater_hi = np.percentile(cheater[cheater > cheater_otsu], 99.5)
        wt_score = (wt.astype(float) - wt_threshold) / max(wt_hi - wt_threshold, 1.0)
        cheater_score = (cheater.astype(float) - cheater_threshold) / max(
            cheater_hi - cheater_threshold, 1.0
        )
        assign_wt = wt_score[overlap] >= cheater_score[overlap]
        wt_mask[overlap] = assign_wt
        cheater_mask[overlap] = ~assign_wt

    metadata = {
        "wt_otsu_threshold": wt_otsu,
        "cheater_otsu_threshold": cheater_otsu,
        "threshold_multiplier": threshold_multiplier,
        "pre_resolution_overlap_fraction": overlap_fraction,
    }
    return wt_mask, cheater_mask, metadata


def xy_neighbor_metric(
    wt_mask: np.ndarray, cheater_mask: np.ndarray, radius_px: int
) -> dict[str, float]:
    """Composition-normalized mixing within square XY neighborhoods.

    For every focal voxel, the opposing-label share among occupied neighbors is
    pooled, then divided by that opposing label's global segmented fraction.
    Total local occupancy is the denominator, so lower total biomass does not
    mechanically lower the score. A random label arrangement is expected near 1.
    """
    occupied = wt_mask | cheater_mask
    side = 2 * radius_px + 1
    area = side * side
    wt_neighbors = (
        ndi.uniform_filter(wt_mask.astype(float), size=(1, side, side), mode="constant")
        * area
        - wt_mask
    )
    cheater_neighbors = (
        ndi.uniform_filter(
            cheater_mask.astype(float), size=(1, side, side), mode="constant"
        )
        * area
        - cheater_mask
    )
    occupied_neighbors = wt_neighbors + cheater_neighbors

    wt_neighbor_total = occupied_neighbors[wt_mask].sum()
    cheater_neighbor_total = occupied_neighbors[cheater_mask].sum()
    raw_wt_to_cheater = cheater_neighbors[wt_mask].sum() / max(wt_neighbor_total, 1)
    raw_cheater_to_wt = wt_neighbors[cheater_mask].sum() / max(
        cheater_neighbor_total, 1
    )

    wt_fraction = wt_mask.sum() / max(occupied.sum(), 1)
    cheater_fraction = cheater_mask.sum() / max(occupied.sum(), 1)
    enrichment_wt_to_cheater = raw_wt_to_cheater / max(cheater_fraction, 1e-12)
    enrichment_cheater_to_wt = raw_cheater_to_wt / max(wt_fraction, 1e-12)
    symmetric = 0.5 * (enrichment_wt_to_cheater + enrichment_cheater_to_wt)
    return {
        "radius_px": radius_px,
        "raw_wt_to_cheater": raw_wt_to_cheater,
        "raw_cheater_to_wt": raw_cheater_to_wt,
        "enrichment_wt_to_cheater": enrichment_wt_to_cheater,
        "enrichment_cheater_to_wt": enrichment_cheater_to_wt,
        "normalized_intermixing": symmetric,
    }


def exact_permutation_p(lb: np.ndarray, treated: np.ndarray) -> float:
    all_values = np.concatenate([lb, treated])
    observed = abs(treated.mean() - lb.mean())
    differences = []
    for treated_idx in itertools.combinations(range(len(all_values)), len(treated)):
        treated_idx = np.array(treated_idx)
        is_treated = np.zeros(len(all_values), dtype=bool)
        is_treated[treated_idx] = True
        differences.append(abs(all_values[is_treated].mean() - all_values[~is_treated].mean()))
    differences = np.asarray(differences)
    return float(np.mean(differences >= observed - 1e-12))


def welch_ci(lb: np.ndarray, treated: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    difference = treated.mean() - lb.mean()
    v1 = treated.var(ddof=1) / len(treated)
    v0 = lb.var(ddof=1) / len(lb)
    se = np.sqrt(v1 + v0)
    if se == 0:
        return difference, difference
    df = (v1 + v0) ** 2 / (
        v1**2 / max(len(treated) - 1, 1) + v0**2 / max(len(lb) - 1, 1)
    )
    critical = stats.t.ppf(1 - alpha / 2, df)
    return float(difference - critical * se), float(difference + critical * se)


def hedges_g(lb: np.ndarray, treated: np.ndarray) -> float:
    df = len(lb) + len(treated) - 2
    pooled = np.sqrt(
        ((len(lb) - 1) * lb.var(ddof=1) + (len(treated) - 1) * treated.var(ddof=1))
        / max(df, 1)
    )
    if pooled == 0:
        return np.nan
    correction = 1 - 3 / max(4 * df - 1, 1)
    return float(correction * (treated.mean() - lb.mean()) / pooled)


def bootstrap_ratio_ci(
    lb: np.ndarray, treated: np.ndarray, seed: int = 9127, draws: int = 20000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    lb_draws = rng.choice(lb, size=(draws, len(lb)), replace=True).mean(axis=1)
    treated_draws = rng.choice(treated, size=(draws, len(treated)), replace=True).mean(axis=1)
    ratios = treated_draws / np.maximum(lb_draws, 1e-12)
    return tuple(float(x) for x in np.percentile(ratios, [2.5, 97.5]))


def add_condition_statistics(
    frame: pd.DataFrame, metric_columns: list[str], id_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    inference_rows: list[dict] = []
    grouped_keys = id_columns
    group_iterator = frame.groupby(grouped_keys, dropna=False) if grouped_keys else [((), frame)]
    for key, group in group_iterator:
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(grouped_keys, key))
        for metric in metric_columns:
            for condition, values in group.groupby("condition")[metric]:
                values = values.astype(float).to_numpy()
                summary_rows.append(
                    {
                        **key_values,
                        "metric": metric,
                        "condition": condition,
                        "n_fields": len(values),
                        "mean": values.mean(),
                        "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                        "minimum": values.min(),
                        "maximum": values.max(),
                    }
                )
            lb = group.loc[group.condition == "LB", metric].astype(float).to_numpy()
            treated = group.loc[group.condition == "STR", metric].astype(float).to_numpy()
            if len(lb) and len(treated):
                difference = treated.mean() - lb.mean()
                ci_low, ci_high = welch_ci(lb, treated)
                ratio_ci_low, ratio_ci_high = bootstrap_ratio_ci(lb, treated)
                inference_rows.append(
                    {
                        **key_values,
                        "metric": metric,
                        "lb_mean": lb.mean(),
                        "str_mean": treated.mean(),
                        "str_minus_lb": difference,
                        "welch_95ci_low": ci_low,
                        "welch_95ci_high": ci_high,
                        "str_over_lb": treated.mean() / max(lb.mean(), 1e-12),
                        "percent_change": 100 * difference / max(lb.mean(), 1e-12),
                        "bootstrap_ratio_95ci_low": ratio_ci_low,
                        "bootstrap_ratio_95ci_high": ratio_ci_high,
                        "hedges_g": hedges_g(lb, treated),
                        "exact_permutation_p_two_sided": exact_permutation_p(lb, treated),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(inference_rows)


def plot_segmentation_qc(records: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(9, 13))
    for ax, record in zip(axes.ravel(), records):
        wt = record["wt"]
        cheater = record["cheater"]
        wt_mask = record["wt_mask"]
        cheater_mask = record["cheater_mask"]
        z = wt.shape[0] // 2
        raw = np.zeros((*wt.shape[1:], 3), dtype=np.uint8)
        raw[..., 1] = wt[z]
        raw[..., 0] = cheater[z]
        ax.imshow(raw)
        contour_wt = ndi.binary_dilation(wt_mask[z]) ^ ndi.binary_erosion(wt_mask[z])
        contour_cheater = ndi.binary_dilation(cheater_mask[z]) ^ ndi.binary_erosion(
            cheater_mask[z]
        )
        overlay = np.zeros((*wt.shape[1:], 4), dtype=float)
        overlay[contour_wt] = (0.1, 1.0, 0.2, 0.8)
        overlay[contour_cheater] = (1.0, 0.2, 0.1, 0.8)
        ax.imshow(overlay)
        ax.set_title(f"{record['sample']} · middle slice z={z + 1}")
        ax.axis("off")
    fig.suptitle("Segmentation QC: raw signal with exclusive-mask boundaries", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_scale(intermixing: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    colors = {"LB": GREEN, "STR": RED}
    for condition in ("LB", "STR"):
        subset = intermixing[intermixing.condition == condition]
        for _, group in subset.groupby("sample"):
            ax.plot(
                group.radius_px,
                group.normalized_intermixing,
                color=colors[condition],
                alpha=0.28,
                linewidth=1.1,
                marker="o",
                markersize=3,
            )
        summary = subset.groupby("radius_px").normalized_intermixing.agg(["mean", "std"])
        ax.plot(
            summary.index,
            summary["mean"],
            color=colors[condition],
            linewidth=2.6,
            marker="o",
            label=f"{condition} mean",
        )
        ax.fill_between(
            summary.index,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=colors[condition],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(1, color=INK, linestyle="--", linewidth=1, label="random-label expectation")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RADII, [str(x) for x in RADII])
    ax.set_xlabel("XY neighborhood radius (pixels)")
    ax.set_ylabel("Composition-normalized intermixing (1 = random mixing)")
    ax.set_title("WT–ΔtnaA intermixing across spatial scales")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_architecture(stack_qc: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    rng = np.random.default_rng(99)
    for ax, metric, ylabel in [
        (axes[0], "apparent_mean_thickness_slices", "Segmented thickness (slices)"),
        (axes[1], "wt_fraction_of_segmented_biomass", "WT fraction of segmented biomass"),
    ]:
        for x, condition in enumerate(["LB", "STR"]):
            values = stack_qc.loc[stack_qc.condition == condition, metric].to_numpy()
            jitter = rng.normal(0, 0.025, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=58,
                color=GREEN if condition == "LB" else RED,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.plot([x - 0.15, x + 0.15], [values.mean(), values.mean()], color=INK, lw=2)
        ax.set_xticks([0, 1], ["LB", "STR"])
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Apparent biofilm amount")
    axes[1].set_title("Population composition")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    effects = []
    for (multiplier, radius), group in sensitivity.groupby(
        ["threshold_multiplier", "radius_px"]
    ):
        lb = group.loc[group.condition == "LB", "normalized_intermixing"].mean()
        treated = group.loc[group.condition == "STR", "normalized_intermixing"].mean()
        effects.append(
            {
                "threshold_multiplier": multiplier,
                "radius_px": radius,
                "percent_change": 100 * (treated - lb) / lb,
            }
        )
    effects = pd.DataFrame(effects)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    shades = {0.8: "#f2aaa5", 1.0: RED, 1.2: "#84221d"}
    for multiplier, group in effects.groupby("threshold_multiplier"):
        ax.plot(
            group.radius_px,
            group.percent_change,
            marker="o",
            linewidth=2,
            color=shades[float(multiplier)],
            label=f"{multiplier:.1f}× Otsu",
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RADII, [str(x) for x in RADII])
    ax.set_xlabel("XY neighborhood radius (pixels)")
    ax.set_ylabel("STR versus LB change in normalized intermixing (%)")
    ax.set_title("Segmentation-threshold sensitivity")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)

    stack_rows: list[dict] = []
    intermixing_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    depth_rows: list[dict] = []
    qc_records: list[dict] = []

    for sample_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        sample = sample_dir.name
        condition, replicate = sample.rsplit("-", 1)
        wt = load_stack(sample_dir / "K12wt", rgb_index=1)
        cheater = load_stack(sample_dir / "K12ΔtnaA", rgb_index=0)
        if wt.shape != cheater.shape:
            raise ValueError(f"Channel shape mismatch in {sample}: {wt.shape} versus {cheater.shape}")

        for multiplier in THRESHOLD_MULTIPLIERS:
            wt_mask, cheater_mask, seg = segment_exclusive(wt, cheater, multiplier)
            occupied = wt_mask | cheater_mask
            for radius in RADII:
                row = {
                    "sample": sample,
                    "condition": condition,
                    "replicate": int(replicate),
                    "threshold_multiplier": multiplier,
                    **xy_neighbor_metric(wt_mask, cheater_mask, radius),
                }
                sensitivity_rows.append(row)
                if multiplier == 1.0:
                    intermixing_rows.append({k: v for k, v in row.items() if k != "threshold_multiplier"})

            if multiplier == 1.0:
                n_xy = occupied.shape[1] * occupied.shape[2]
                stack_rows.append(
                    {
                        "sample": sample,
                        "condition": condition,
                        "replicate": int(replicate),
                        "z_slices": occupied.shape[0],
                        "height_px": occupied.shape[1],
                        "width_px": occupied.shape[2],
                        "wt_otsu_threshold": seg["wt_otsu_threshold"],
                        "cheater_otsu_threshold": seg["cheater_otsu_threshold"],
                        "pre_resolution_overlap_fraction": seg[
                            "pre_resolution_overlap_fraction"
                        ],
                        "wt_segmented_voxels": int(wt_mask.sum()),
                        "cheater_segmented_voxels": int(cheater_mask.sum()),
                        "total_segmented_voxels": int(occupied.sum()),
                        "wt_volume_fraction": wt_mask.mean(),
                        "cheater_volume_fraction": cheater_mask.mean(),
                        "total_volume_fraction": occupied.mean(),
                        "apparent_mean_thickness_slices": occupied.sum() / n_xy,
                        "wt_apparent_thickness_slices": wt_mask.sum() / n_xy,
                        "cheater_apparent_thickness_slices": cheater_mask.sum() / n_xy,
                        "wt_fraction_of_segmented_biomass": wt_mask.sum()
                        / max(occupied.sum(), 1),
                        "cheater_fraction_of_segmented_biomass": cheater_mask.sum()
                        / max(occupied.sum(), 1),
                        "wt_mean_intensity": wt.mean(),
                        "cheater_mean_intensity": cheater.mean(),
                    }
                )
                qc_records.append(
                    {
                        "sample": sample,
                        "wt": wt,
                        "cheater": cheater,
                        "wt_mask": wt_mask,
                        "cheater_mask": cheater_mask,
                    }
                )
                for band_index, z_indices in enumerate(np.array_split(np.arange(wt.shape[0]), 3), 1):
                    band_wt = wt_mask[z_indices]
                    band_cheater = cheater_mask[z_indices]
                    for radius in (4, 8):
                        depth_rows.append(
                            {
                                "sample": sample,
                                "condition": condition,
                                "replicate": int(replicate),
                                "relative_depth_band": ["bottom", "middle", "top"][band_index - 1],
                                "radius_px": radius,
                                **xy_neighbor_metric(band_wt, band_cheater, radius),
                            }
                        )

    stack_qc = pd.DataFrame(stack_rows).sort_values(["condition", "replicate"])
    intermixing = pd.DataFrame(intermixing_rows).sort_values(
        ["condition", "replicate", "radius_px"]
    )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["threshold_multiplier", "condition", "replicate", "radius_px"]
    )
    depth = pd.DataFrame(depth_rows).sort_values(
        ["relative_depth_band", "radius_px", "condition", "replicate"]
    )

    stack_summary, stack_inference = add_condition_statistics(
        stack_qc,
        [
            "apparent_mean_thickness_slices",
            "wt_fraction_of_segmented_biomass",
            "cheater_fraction_of_segmented_biomass",
        ],
        [],
    )
    mixing_summary, mixing_inference = add_condition_statistics(
        intermixing,
        ["normalized_intermixing"],
        ["radius_px"],
    )
    depth_summary, depth_inference = add_condition_statistics(
        depth,
        ["normalized_intermixing"],
        ["relative_depth_band", "radius_px"],
    )

    for name, frame in {
        "stack_qc.csv": stack_qc,
        "intermixing_by_radius.csv": intermixing,
        "threshold_sensitivity.csv": sensitivity,
        "relative_depth.csv": depth,
        "condition_summary.csv": pd.concat(
            [stack_summary.assign(analysis="architecture"), mixing_summary.assign(analysis="intermixing")],
            ignore_index=True,
        ),
        "condition_inference.csv": pd.concat(
            [stack_inference.assign(analysis="architecture"), mixing_inference.assign(analysis="intermixing")],
            ignore_index=True,
        ),
        "relative_depth_summary.csv": depth_summary,
        "relative_depth_inference.csv": depth_inference,
    }.items():
        frame.to_csv(args.output / name, index=False)

    plot_segmentation_qc(qc_records, figure_dir / "segmentation_qc.png")
    plot_scale(intermixing, figure_dir / "intermixing_by_scale.png")
    plot_architecture(stack_qc, figure_dir / "biofilm_amount_and_composition.png")
    plot_threshold_sensitivity(sensitivity, figure_dir / "threshold_sensitivity.png")


if __name__ == "__main__":
    main()
