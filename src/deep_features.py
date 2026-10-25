"""Frozen ImageNet ResNet18 embeddings and train-only PCA pair features.

Original RGB images are read-only. Model inference, PCA, and pair construction
never use filenames or orchard IDs as numerical features. No CNN fine-tuning.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from .features import stable_digest
from .utils import file_digest, read_rgb


def download_weights(root: Path, settings: dict) -> Path:
    """Download and verify the explicitly pinned official ResNet18 checkpoint.

    Args:
        root (Path): Project base for the checkpoint path.
        settings (dict): checkpoint, pinned weights_url, weights name and hash prefix.

    Returns:
        Path: Verified local checkpoint path; an existing valid file is reused.

    Raises:
        ValueError: The URL, weight name or digest differs from the pinned version.
        OSError: Network download or local checkpoint access fails.
    """
    if settings["weights_url"] != "https://download.pytorch.org/models/resnet18-f37072fd.pth":
        raise ValueError("Only the pinned official ResNet18 checkpoint is supported")
    path = Path(root) / settings["checkpoint"]
    if path.exists():
        verify_weights(path, settings)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    urllib.request.urlretrieve(settings["weights_url"], temporary)
    verify_weights(temporary, settings)
    temporary.replace(path)
    return path


def verify_weights(path: Path, settings: dict) -> str:
    """Check the required weight version and file digest before encoder loading.

    Args:
        path (Path): Checkpoint file to verify.
        settings (dict): weights="IMAGENET1K_V1" and weights_hash_prefix="f37072fd".

    Returns:
        str: Complete SHA-256 digest whose prefix matches the official checkpoint.

    Raises:
        FileNotFoundError: The checkpoint is absent.
        ValueError: Weight identifiers or checksum prefix do not match.
        OSError: Checkpoint bytes cannot be read.
    """
    if settings["weights"] != "IMAGENET1K_V1" or settings["weights_hash_prefix"] != "f37072fd":
        raise ValueError("Use the pinned IMAGENET1K_V1 weights")
    if not path.is_file():
        raise FileNotFoundError("Checkpoint missing; run tools/download_resnet_weights.py first")
    digest = file_digest(path)
    if not digest.startswith(settings["weights_hash_prefix"]):
        raise ValueError("ResNet18 checkpoint checksum does not match the official weights")
    return digest


def frozen_encoder(checkpoint: Path | None, device: str = "cpu"):
    """Construct a frozen, evaluation-mode ResNet18 with a 512-dimensional output.

    Args:
        checkpoint (Path | None): Verified state dictionary; None uses random weights for
            architecture tests only.
        device (str): PyTorch target device such as "cpu" or "cuda".

    Returns:
        torch.nn.Module: Encoder with its final classifier replaced by Identity and gradients
        disabled.

    Raises:
        OSError: The checkpoint cannot be read.
        RuntimeError: State-dictionary loading or device placement fails.
    """
    model = resnet18(weights=None)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    # Remove class logits so downstream PCA receives the frozen image representation.
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    return model.to(device).eval()


def _unit_vectors(values):
    """Validate and L2-normalize a batch of nonzero image embeddings.

    Args:
        values (array-like): Finite embedding matrix with shape (N, 512).

    Returns:
        numpy.ndarray: Float64 matrix with unit row norms and shape (N, 512).

    Raises:
        ValueError: Shape, finiteness or a nonzero row-norm requirement fails.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonzero N×512 image embeddings")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Expected finite, nonzero N×512 image embeddings")
    return values / norm


def embed_paths(paths, root: Path, settings: dict) -> tuple[dict, dict]:
    """Embed explicit RGB paths using a verified frozen encoder and numeric content caches.

    A centered square uses official V1 resize/crop/normalization. Keys include
    image/weight/source hashes, software versions, device and deterministic runtime
    settings. The caller controls which split paths reach this low-level function.

    Args:
        paths (iterable[str | Path]): RGB paths, absolute or relative to root; duplicates are
            embedded once.
        root (Path): Project base for images, checkpoint and caches.
        settings (dict): Verified weight fields, checkpoint/cache paths, device, batch_size,
            cpu_threads and seed.

    Returns:
        tuple[dict, dict]: Mapping from string paths to normalized 512-element vectors, and
        encoder/cache/device/version/timing provenance.

    Raises:
        FileNotFoundError: The checkpoint or an RGB file is absent.
        ValueError: Weights, RGB data, cached embeddings or device/batch/thread settings are
            invalid.
        RuntimeError: PyTorch inference or device allocation fails.
        OSError: Image/checkpoint/cache access fails.
    """
    start = perf_counter()
    checkpoint = root / settings["checkpoint"]
    weights_sha = verify_weights(checkpoint, settings)
    if settings["device"] not in {"auto", "cpu", "cuda"} or settings["batch_size"] < 1 or settings["cpu_threads"] < 1:
        raise ValueError("Invalid device, batch size, or CPU thread count")
    device = "cuda" if settings["device"] == "auto" and torch.cuda.is_available() else settings["device"]
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}
    signature = stable_digest({"weights_sha256": weights_sha, "versions": versions,
        "code_sha256": file_digest(Path(__file__)), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "batch_size": settings["batch_size"], "cpu_threads": settings["cpu_threads"],
        "preprocessing": "center_square_then_V1_resize256_crop224_normalize"})[:20]
    folder = root / settings["cache_dir"] / signature
    folder.mkdir(parents=True, exist_ok=True)
    results, missing = {}, []
    for path in sorted(set(map(str, paths))):
        digest = file_digest(root / path)
        cache = folder / f"{digest}.npz"
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                embedding = saved["embedding"]
            _unit_vectors(embedding.reshape(1, -1))
            results[path] = embedding.astype(np.float64)
        else:
            missing.append((path, cache))
    if missing:
        model = frozen_encoder(checkpoint, device)
        transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        for start_index in range(0, len(missing), settings["batch_size"]):
            batch = missing[start_index:start_index+settings["batch_size"]]
            tensors = []
            for path, _ in batch:
                rgb = read_rgb(root / path)
                h, w = rgb.shape[:2]
                side = min(h, w)
                rgb = rgb[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
                tensors.append(transform(Image.fromarray(rgb)))
            with torch.inference_mode():
                output = model(torch.stack(tensors).to(device)).cpu().numpy()
            output = _unit_vectors(output)
            for (path, cache), vector in zip(batch, output):
                np.savez_compressed(cache, embedding=vector)
                results[path] = vector
            completed = min(start_index+len(batch), len(missing))
            if completed % 128 < settings["batch_size"] or completed == len(missing):
                print(f"Frozen ResNet18: {completed}/{len(missing)} new images ({device})", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return results, {"image_count": len(results), "cache_hits": len(results)-len(missing),
                     "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                     "weights_sha256": weights_sha, "weights_url": settings["weights_url"], "weights": settings["weights"],
                     "signature": signature, "versions": versions, "seconds": perf_counter()-start,
                     "cnn_parameters_trained": 0}


def fit_projection(embeddings: pd.DataFrame, components=32) -> PCA:
    """Fit PCA using both seasons of training images only.

    Args:
        embeddings (pandas.DataFrame): Rows with split and finite nonzero 512-element embedding
            arrays.
        components (int): Positive PCA dimension, at most 512 and smaller than the number of
            training images.

    Returns:
        sklearn.decomposition.PCA: Fitted full-SVD projection; validation rows never enter fit.

    Raises:
        ValueError: Component count or training embedding dimensions/values are invalid.
    """
    # Both training seasons may define PCA; held-out images may only be transformed.
    train = embeddings[embeddings.split.eq("train")]
    if not isinstance(components, int) or components < 1 or components >= min(len(train), 513):
        raise ValueError("PCA components must be positive and smaller than the number of training images")
    matrix = np.stack(train.embedding)
    _unit_vectors(matrix)
    return PCA(n_components=components, svd_solver="full").fit(matrix)


def pair_columns(components: int) -> list[str]:
    """Name PCA-coordinate differences, products and original-vector cosine distance.

    Args:
        components (int): Number of fitted PCA coordinates.

    Returns:
        list[str]: Ordered 2 * components + 1 feature names.
    """
    return ([f"pca_abs_{i:02d}" for i in range(components)] +
            [f"pca_product_{i:02d}" for i in range(components)] + ["deep_cosine_distance"])


def pair_matrix(query_embeddings, candidate_embeddings, projection):
    """Construct dimensionless PCA differences/products and embedding cosine distances.

    Args:
        query_embeddings (array-like): Nonzero finite query vectors with shape (N, 512).
        candidate_embeddings (array-like): Aligned candidate vectors with the same shape.
        projection (sklearn.decomposition.PCA): Projection fitted on training images only.

    Returns:
        numpy.ndarray: Shape (N, 2 * components + 1), ordered as absolute PCA differences, PCA
        products, then cosine distance clipped to [0, 2].

    Raises:
        ValueError: Embedding shapes/values or fitted projection dimensions are incompatible.
    """
    query, candidate = _unit_vectors(query_embeddings), _unit_vectors(candidate_embeddings)
    if query.shape != candidate.shape:
        raise ValueError("Pair arrays must have equal shapes")
    q, c = projection.transform(query), projection.transform(candidate)
    cosine_distance = np.clip(1-np.einsum("ij,ij->i", query, candidate), 0, 2)
    return np.c_[np.abs(q-c), q*c, cosine_distance]


def build_pair_tables(embeddings: pd.DataFrame, projection) -> dict[str, pd.DataFrame]:
    """Construct complete train/validation pair matrices from seasonal embeddings.

    Args:
        embeddings (pandas.DataFrame): Development tree_id/split/season/embedding rows with one
            Summer and Winter image per identity.
        projection (sklearn.decomposition.PCA): Training-fitted PCA projection.

    Returns:
        dict[str, pandas.DataFrame]: Complete split galleries with split/query_id/candidate_id
        metadata and numeric columns from pair_columns.

    Raises:
        ValueError: Test observations, duplicated/missing seasons or invalid embeddings are
            supplied.
    """
    if not set(embeddings.split).issubset({"train", "validation"}):
        raise ValueError("Development embeddings must exclude test observations")
    columns = pair_columns(projection.n_components_)
    frames = {}
    for split, group in embeddings.groupby("split", sort=False):
        query = group[group.season.eq("Summer")].sort_values("tree_id")
        candidate = group[group.season.eq("Winter")].sort_values("tree_id")
        if (query.empty or query.tree_id.duplicated().any() or candidate.tree_id.duplicated().any() or
                query.tree_id.tolist() != candidate.tree_id.tolist()):
            raise ValueError("Each development identity needs one image per season")
        n = len(query)
        q, c = _unit_vectors(np.stack(query.embedding)), _unit_vectors(np.stack(candidate.embedding))
        qp, cp = projection.transform(q), projection.transform(c)
        values = np.c_[np.abs(qp[:, None, :]-cp[None, :, :]).reshape(n*n, -1),
                       (qp[:, None, :]*cp[None, :, :]).reshape(n*n, -1),
                       np.clip(1-q@c.T, 0, 2).ravel()]
        ids = pd.DataFrame({"split": split, "query_id": np.repeat(query.tree_id.to_numpy(), n),
                            "candidate_id": np.tile(candidate.tree_id.to_numpy(), n)})
        frames[split] = pd.concat([ids, pd.DataFrame(values, columns=columns)], axis=1)
    return frames
