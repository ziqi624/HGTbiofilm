#!/usr/bin/env python3

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

GAUSSIAN_SIGMA = (0.0, 1.0, 1.0)  # no smoothing across z
MIN_OBJECT_VOXELS = 8


def z_number(path: Path) -> int:
    match = re.search(r"_z(\d+)", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse z index from {path.name}")
    return int(match.group(1))


def identify_files(rep_dir: Path, role: str) -> list[Path]:
    token = "(donor)" if role == "donor" else "(recipient"
    files = [p for p in rep_dir.rglob("*.tif") if token in str(p).lower()]
    return sorted(files, key=z_number)


def read_signal_stack(files: list[Path], rgb_plane: int) -> np.ndarray:
    images = []
    for path in files:
        image = np.asarray(Image.open(path))
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image[..., rgb_plane]
        if image.ndim != 2:
            raise ValueError(f"Unsupported image shape {image.shape}: {path}")
        images.append(image)
    return np.stack(images).astype(np.uint8, copy=False)


def threshold_otsu(image: np.ndarray, nbins: int = 256) -> float:
    """Equivalent to skimage.filters.threshold_otsu for a float-valued image."""
    values = np.asarray(image).ravel()
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        return lo
    counts, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((counts * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    return float(centers[int(np.argmax(variance12))])


def remove_small_objects(mask: np.ndarray, max_size: int, connectivity: int = 1) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labels, _ = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    too_small = sizes <= max_size
    too_small[0] = False
    cleaned = mask.copy()
    cleaned[too_small[labels]] = False
    return cleaned


def segment(stack: np.ndarray, multiplier: float = 1.0) -> tuple[np.ndarray, float]:
    smooth = gaussian_filter(stack.astype(np.float32), sigma=GAUSSIAN_SIGMA)
    sample = smooth.ravel()[::4]
    threshold = float(threshold_otsu(sample)) * multiplier
    mask = smooth > threshold
    mask = remove_small_objects(mask, max_size=MIN_OBJECT_VOXELS - 1, connectivity=1)
    return mask, threshold


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _validate_mask_pair(
    donor_mask: np.ndarray, recipient_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Boolean masks after checking that their 3D shapes match."""
    donor_mask = np.asarray(donor_mask, dtype=bool)
    recipient_mask = np.asarray(recipient_mask, dtype=bool)
    if donor_mask.ndim != 3 or recipient_mask.ndim != 3:
        raise ValueError("Donor and recipient masks must both be 3D (z, y, x).")
    if donor_mask.shape != recipient_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: {donor_mask.shape} vs {recipient_mask.shape}"
        )
    return donor_mask, recipient_mask


def jaccard_3d(donor_mask: np.ndarray, recipient_mask: np.ndarray) -> float:
    """Calculate the 3D Jaccard index from two segmented Boolean masks.

    Formula: ``|D intersection R| / |D union R|``.

    A value of 0 means no same-voxel overlap and 1 means identical positive
    masks. ``NaN`` is returned when both masks are empty and their union is 0.
    """
    donor_mask, recipient_mask = _validate_mask_pair(donor_mask, recipient_mask)
    intersection_voxels = int(np.count_nonzero(donor_mask & recipient_mask))
    union_voxels = int(np.count_nonzero(donor_mask | recipient_mask))
    return safe_ratio(intersection_voxels, union_voxels)


def donor_voxel_fraction_with_recipient(
    donor_mask: np.ndarray, recipient_mask: np.ndarray
) -> float:
    """Fraction of donor-positive voxels that are also recipient-positive.

    Formula: ``|D intersection R| / |D|``. This is a directional binary-mask
    measurement. ``NaN`` is returned when the donor mask contains no positive
    voxels.
    """
    donor_mask, recipient_mask = _validate_mask_pair(donor_mask, recipient_mask)
    intersection_voxels = int(np.count_nonzero(donor_mask & recipient_mask))
    donor_voxels = int(np.count_nonzero(donor_mask))
    return safe_ratio(intersection_voxels, donor_voxels)


def recipient_voxel_fraction_with_donor(
    donor_mask: np.ndarray, recipient_mask: np.ndarray
) -> float:
    """Fraction of recipient-positive voxels that are also donor-positive.

    Formula: ``|D intersection R| / |R|``. This is a directional binary-mask
    measurement. ``NaN`` is returned when the recipient mask is empty.
    """
    donor_mask, recipient_mask = _validate_mask_pair(donor_mask, recipient_mask)
    intersection_voxels = int(np.count_nonzero(donor_mask & recipient_mask))
    recipient_voxels = int(np.count_nonzero(recipient_mask))
    return safe_ratio(intersection_voxels, recipient_voxels)


def manders_donor_in_recipient(
    donor: np.ndarray,
    donor_mask: np.ndarray,
    recipient_mask: np.ndarray,
    donor_threshold: float,
) -> float:
    """Thresholded Manders fraction of donor intensity in the mask intersection.

    Donor intensity is background-subtracted as
    ``max(donor - donor_threshold, 0)``. The numerator is this donor intensity
    inside ``D intersection R``; the denominator is donor intensity inside D.
    ``NaN`` is returned if the segmented donor intensity is 0.
    """
    donor_mask, recipient_mask = _validate_mask_pair(donor_mask, recipient_mask)
    donor = np.asarray(donor)
    if donor.shape != donor_mask.shape:
        raise ValueError(f"Donor image/mask shape mismatch: {donor.shape} vs {donor_mask.shape}")
    intersection = donor_mask & recipient_mask
    donor_signal = np.clip(donor.astype(np.float32) - donor_threshold, 0, None)
    donor_total = float(donor_signal[donor_mask].sum())
    donor_overlap = float(donor_signal[intersection].sum())
    return safe_ratio(donor_overlap, donor_total)


def manders_recipient_in_donor(
    recipient: np.ndarray,
    donor_mask: np.ndarray,
    recipient_mask: np.ndarray,
    recipient_threshold: float,
) -> float:
    """Thresholded Manders fraction of recipient intensity in the intersection.

    Recipient intensity is background-subtracted as
    ``max(recipient - recipient_threshold, 0)``. The numerator is this
    recipient intensity inside ``D intersection R``; the denominator is
    recipient intensity inside R. ``NaN`` is returned if that total is 0.
    """
    donor_mask, recipient_mask = _validate_mask_pair(donor_mask, recipient_mask)
    recipient = np.asarray(recipient)
    if recipient.shape != recipient_mask.shape:
        raise ValueError(
            f"Recipient image/mask shape mismatch: {recipient.shape} vs {recipient_mask.shape}"
        )
    intersection = donor_mask & recipient_mask
    recipient_signal = np.clip(
        recipient.astype(np.float32) - recipient_threshold, 0, None
    )
    recipient_total = float(recipient_signal[recipient_mask].sum())
    recipient_overlap = float(recipient_signal[intersection].sum())
    return safe_ratio(recipient_overlap, recipient_total)


def colocalization_metrics(
    donor: np.ndarray,
    recipient: np.ndarray,
    donor_mask: np.ndarray,
    recipient_mask: np.ndarray,
    donor_threshold: float,
    recipient_threshold: float,
) -> dict[str, float | int]:
    intersection = donor_mask & recipient_mask
    union = donor_mask | recipient_mask
    n_donor = int(donor_mask.sum())
    n_recipient = int(recipient_mask.sum())
    n_intersection = int(intersection.sum())
    n_union = int(union.sum())

    donor_signal = np.clip(donor.astype(np.float32) - donor_threshold, 0, None)
    recipient_signal = np.clip(recipient.astype(np.float32) - recipient_threshold, 0, None)
    if n_union > 1 and np.std(donor_signal[union]) > 0 and np.std(recipient_signal[union]) > 0:
        pearson_union = float(pearsonr(donor_signal[union], recipient_signal[union]).statistic)
    else:
        pearson_union = math.nan

    return {
        "total_voxels": int(donor.size),
        "donor_positive_voxels": n_donor,
        "recipient_positive_voxels": n_recipient,
        "intersection_voxels": n_intersection,
        "union_voxels": n_union,
        "donor_volume_fraction": safe_ratio(n_donor, donor.size),
        "recipient_volume_fraction": safe_ratio(n_recipient, recipient.size),
        "jaccard_3d": jaccard_3d(donor_mask, recipient_mask),
        "dice_3d": safe_ratio(2 * n_intersection, n_donor + n_recipient),
        "overlap_coefficient_3d": safe_ratio(n_intersection, min(n_donor, n_recipient)),
        "donor_voxel_fraction_with_recipient": donor_voxel_fraction_with_recipient(
            donor_mask, recipient_mask
        ),
        "recipient_voxel_fraction_with_donor": recipient_voxel_fraction_with_donor(
            donor_mask, recipient_mask
        ),
        "manders_donor_in_recipient": manders_donor_in_recipient(
            donor, donor_mask, recipient_mask, donor_threshold
        ),
        "manders_recipient_in_donor": manders_recipient_in_donor(
            recipient, donor_mask, recipient_mask, recipient_threshold
        ),
        "pearson_union_bg_subtracted": pearson_union,
    }


def main() -> None:
    records = []

    for rep_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        match = re.fullmatch(r"wt-LB-([13])d-(\d+)", rep_dir.name)
        if not match:
            continue
        day_number, replicate = int(match.group(1)), int(match.group(2))
        day = f"Day {day_number}"

        donor_files = identify_files(rep_dir, "donor")
        recipient_files = identify_files(rep_dir, "recipient")
        if len(donor_files) != len(recipient_files):
            raise ValueError(f"Unmatched channels in {rep_dir.name}")
        if [z_number(p) for p in donor_files] != [z_number(p) for p in recipient_files]:
            raise ValueError(f"Unmatched z indices in {rep_dir.name}")

        donor = read_signal_stack(donor_files, rgb_plane=1)
        recipient = read_signal_stack(recipient_files, rgb_plane=0)
        if donor.shape != recipient.shape:
            raise ValueError(f"Shape mismatch in {rep_dir.name}: {donor.shape} vs {recipient.shape}")

        donor_mask, donor_threshold = segment(donor)
        recipient_mask, recipient_threshold = segment(recipient)
        metrics = colocalization_metrics(
            donor,
            recipient,
            donor_mask,
            recipient_mask,
            donor_threshold,
            recipient_threshold,
        )
        records.append(
            {
                "strain": "wild type",
                "condition_code": "L",
                "condition": "LB",
                "day": day,
                "day_number": day_number,
                "replicate": replicate,
                "sample_id": rep_dir.name,
                "z_slices": donor.shape[0],
                "height_px": donor.shape[1],
                "width_px": donor.shape[2],
                "donor_threshold_otsu_smoothed": donor_threshold,
                "recipient_threshold_otsu_smoothed": recipient_threshold,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(records).sort_values(["day_number", "replicate"])

    numeric_summary = [
        "jaccard_3d",
        "donor_voxel_fraction_with_recipient",
        "recipient_voxel_fraction_with_donor",
        "manders_donor_in_recipient",
        "manders_recipient_in_donor",
    ]
    summary_rows = []
    for day, group in metrics_df.groupby("day", sort=False):
        for metric in numeric_summary:
            values = group[metric].dropna()
            summary_rows.append(
                {
                    "day": day,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    metrics_df.to_csv(OUT / "wt_LB_donor_recipient_replicate_metrics.csv", index=False)
    summary_df.to_csv(OUT / "wt_LB_donor_recipient_day_summary.csv", index=False)

    print("Replicate-level metrics:")
    print(metrics_df.to_string(index=False))
    print("\nDay-level summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
