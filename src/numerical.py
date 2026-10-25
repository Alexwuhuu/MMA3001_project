"""Stage 4: affine least squares and independently checked robust fitting."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import write_json


def fit_affine_lstsq(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve a six-parameter affine model in float64 with a full-rank source design.

    Args:
        source (array-like): Finite source pixel coordinates, shape (N, 2), N >= 3.
        target (array-like): Matching finite destination coordinates with the same shape.

    Returns:
        tuple[numpy.ndarray, dict]: A (2, 3) affine matrix and rank, condition_number and
        fit_rmse_px diagnostics.

    Raises:
        ValueError: Shapes disagree, values are non-finite, or source points are
            collinear/repeated.
        numpy.linalg.LinAlgError: The least-squares decomposition fails.

    Examples:
        >>> import numpy as np
        >>> points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> matrix, diagnostics = fit_affine_lstsq(points, points + [2., 3.])
        >>> np.allclose(transform_points(points, matrix), points + [2., 3.])
        True
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape or len(source) < 3:
        raise ValueError("Affine fitting requires matching N×2 arrays with at least three points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Coordinates must be finite")
    design = np.c_[source, np.ones(len(source))]
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank != 3:
        raise ValueError("Degenerate affine fit: source points are collinear or repeated")
    residual = design @ coefficients - target
    return coefficients.T, {"rank": int(rank), "condition_number": float(singular[0] / singular[-1]),
                            "fit_rmse_px": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))}


def transform_points(source: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine matrix using homogeneous source coordinates.

    Args:
        source (array-like): Pixel coordinates with shape (N, 2).
        matrix (array-like): Affine transform with shape (2, 3).

    Returns:
        numpy.ndarray: Transformed pixel coordinates with shape (N, 2).

    Raises:
        ValueError: Coordinates or matrix dimensions are incompatible.
    """
    return np.c_[source, np.ones(len(source))] @ np.asarray(matrix).T


def fit_affine_ransac(source, target, threshold=3.0, seed=42):
    """Select affine inliers with RANSAC and refit them using the shared least-squares solver.

    Args:
        source (array-like): Finite source coordinates with shape (N, 2), N >= 3.
        target (array-like): Matching destination coordinates.
        threshold (float): Positive RANSAC reprojection threshold in pixels.
        seed (int): Seed assigned to the OpenCV random generator.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, dict]: Refit (2, 3) matrix, Boolean inlier mask of
        length N, and least-squares rank/condition/RMSE diagnostics.

    Raises:
        ValueError: Input validation fails or RANSAC cannot produce three usable inliers.
        cv2.error: RANSAC options are invalid.
    """
    # Reject degenerate inputs before OpenCV can choose an arbitrary affine model.
    fit_affine_lstsq(source, target)
    cv2.setRNGSeed(seed)
    matrix, mask = cv2.estimateAffine2D(np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64),
                                      method=cv2.RANSAC, ransacReprojThreshold=threshold,
                                      maxIters=2000, confidence=0.99, refineIters=10)
    if matrix is None or mask is None or int(mask.sum()) < 3:
        raise ValueError("RANSAC could not estimate an affine transform")
    inliers = mask.ravel().astype(bool)
    matrix, diagnostics = fit_affine_lstsq(np.asarray(source)[inliers], np.asarray(target)[inliers])
    return matrix, inliers, diagnostics


def run_numerical_verification(output: str | Path, seed: int = 42, render: bool = True) -> dict:
    """Verify recovery on independent probe points under controlled noise and outliers.

    Probe points never influence fitting or inlier selection. These checks establish
    synthetic pixel-coordinate behavior, without validating seasonal identities.

    Args:
        output (str | Path): Directory for experiment CSVs, summary JSON and optional figure.
        seed (int): Seed for synthetic coordinates, noise and RANSAC repetitions.
        render (bool): Whether to write affine_verification.png.

    Returns:
        dict: summary, experiments, summary_table and output_dir for the completed verification.

    Raises:
        AssertionError: Noiseless probe RMSE is at least 1e-6 pixels.
        ValueError: A synthetic affine fit fails validation.
        OSError: Verification artifacts cannot be written.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = np.array([[1.03, -0.12, 11.0], [0.08, 0.96, -7.0]], dtype=np.float64)
    source = rng.uniform(20, 220, (100, 2))
    probe = rng.uniform(10, 245, (200, 2))
    clean_target, probe_truth = transform_points(source, truth), transform_points(probe, truth)
    exact, exact_diagnostics = fit_affine_lstsq(source, clean_target)
    exact_error = float(np.sqrt(np.mean(np.sum((transform_points(probe, exact) - probe_truth) ** 2, axis=1))))
    if exact_error >= 1e-6:
        raise AssertionError(f"Noiseless recovery failed: {exact_error} px")
    records = []
    illustration = None
    for noise in [0.0, 0.5, 1.0, 2.0]:
        for outlier_fraction in [0.0, 0.1, 0.25, 0.4]:
            for repetition in range(5):
                target = clean_target + rng.normal(0, noise, clean_target.shape)
                is_outlier = np.zeros(len(source), dtype=bool)
                if outlier_fraction:
                    indices = rng.choice(len(source), round(len(source) * outlier_fraction), replace=False)
                    is_outlier[indices] = True
                    target[indices] = rng.uniform(0, 256, (len(indices), 2))
                for method in ["all_correspondences_lstsq", "ransac_inliers_lstsq"]:
                    started = perf_counter()
                    if method == "all_correspondences_lstsq":
                        matrix, diagnostics = fit_affine_lstsq(source, target)
                        selected = np.ones(len(source), dtype=bool)
                    else:
                        matrix, selected, diagnostics = fit_affine_ransac(source, target, seed=seed + repetition)
                    elapsed = perf_counter() - started
                    errors = np.linalg.norm(transform_points(probe, matrix) - probe_truth, axis=1)
                    records.append({"method": method, "noise_sigma_px": noise, "outlier_fraction": outlier_fraction,
                                    "repetition": repetition, "probe_rmse_px": float(np.sqrt(np.mean(errors ** 2))),
                                    "parameter_error_frobenius": float(np.linalg.norm(matrix - truth)),
                                    "selected_count": int(selected.sum()),
                                    "selected_true_inlier_fraction": float((~is_outlier[selected]).mean()),
                                    "fit_seconds": elapsed, **diagnostics})
                    if noise == 0.5 and outlier_fraction == 0.25 and repetition == 0:
                        if illustration is None:
                            illustration = {"source": source, "target": target, "probe": probe, "truth": probe_truth}
                        illustration[method] = transform_points(probe, matrix)
    results = pd.DataFrame(records)
    results.to_csv(output / "affine_experiments.csv", index=False)
    summary_table = results.groupby(["method", "noise_sigma_px", "outlier_fraction"], as_index=False).agg(
        median_probe_rmse_px=("probe_rmse_px", "median"), median_fit_seconds=("fit_seconds", "median"))
    summary_table.to_csv(output / "affine_summary.csv", index=False)
    selected = results[(results.noise_sigma_px.eq(0.5)) & results.outlier_fraction.eq(0.25)]
    medians = selected.groupby("method").probe_rmse_px.median().to_dict()
    summary = {"noiseless_probe_rmse_px": exact_error, "noiseless_acceptance_px": 1e-6,
               "noiseless_passed": True, "exact_fit_diagnostics": exact_diagnostics,
               "experiment_rows": len(results), "repetitions_per_setting": 5,
               "probe_points_not_used_for_fitting": len(probe),
               "median_probe_rmse_at_25pct_outliers_sigma_0_5": medians,
               "robust_improvement_observed": medians["ransac_inliers_lstsq"] < medians["all_correspondences_lstsq"],
               "units": "pixels in a synthetic 256×256 coordinate domain", "seed": seed}
    if render:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        subset = summary_table[summary_table.noise_sigma_px.eq(0.5)]
        for name, group in subset.groupby("method"):
            axes[0].plot(group.outlier_fraction, group.median_probe_rmse_px, "o-", label=name)
        axes[0].set(xlabel="Outlier fraction", ylabel="Independent-probe RMSE (px)", title="Sensitivity to incorrect correspondences")
        axes[0].legend(fontsize=8)
        take = np.arange(0, len(probe), 10)
        axes[1].scatter(probe_truth[take, 0], probe_truth[take, 1], marker="+", color="black", label="Known transform")
        for method, color in [("all_correspondences_lstsq", "#c65353"), ("ransac_inliers_lstsq", "#327fab")]:
            values = illustration[method][take]
            axes[1].scatter(values[:, 0], values[:, 1], s=18, color=color, label=method)
        axes[1].set(xlabel="Transformed x (px)", ylabel="Transformed y (px)", title="25% outliers, noise sigma=0.5 px")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "affine_verification.png", dpi=130)
        plt.close(fig)
    write_json(output / "verification_summary.json", summary)
    return {"summary": summary, "experiments": results, "summary_table": summary_table, "output_dir": output}
