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
    """Download the explicitly pinned official checkpoint and verify its hash prefix."""
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
    if settings["weights"] != "IMAGENET1K_V1" or settings["weights_hash_prefix"] != "f37072fd":
        raise ValueError("Use the pinned IMAGENET1K_V1 weights")
    if not path.is_file():
        raise FileNotFoundError("Checkpoint missing; run tools/download_resnet_weights.py first")
    digest = file_digest(path)
    if not digest.startswith(settings["weights_hash_prefix"]):
        raise ValueError("ResNet18 checkpoint checksum does not match the official weights")
    return digest


def frozen_encoder(checkpoint: Path | None, device: str = "cpu"):
    """Build a frozen, evaluation-mode 512-D encoder; None is for architecture tests only."""
    model = resnet18(weights=None)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    return model.to(device).eval()


def _unit_vectors(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 512 or not np.isfinite(values).all():
        raise ValueError("Expected finite, nonzero N×512 image embeddings")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("Expected finite, nonzero N×512 image embeddings")
    return values / norm


def embed_paths(paths, root: Path, settings: dict) -> tuple[dict, dict]:
    """Embed only the explicit RGB paths, with safe numeric caches and content keys.

    Preprocessing: center-square crop, then the official V1 resize256/crop224/
    ImageNet normalization. Cache keys include image/weight/code hashes, versions,
    device, batch size, and deterministic inference settings. Test paths are never
    supplied by the development workflow. CPU is supported when CUDA is absent.
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
    """Fit PCA on both seasons of training images only, with no row/identity inputs."""
    train = embeddings[embeddings.split.eq("train")]
    if not isinstance(components, int) or components < 1 or components >= min(len(train), 513):
        raise ValueError("PCA components must be positive and smaller than the number of training images")
    matrix = np.stack(train.embedding)
    _unit_vectors(matrix)
    return PCA(n_components=components, svd_solver="full").fit(matrix)


def pair_columns(components: int) -> list[str]:
    return ([f"pca_abs_{i:02d}" for i in range(components)] +
            [f"pca_product_{i:02d}" for i in range(components)] + ["deep_cosine_distance"])


def pair_matrix(query_embeddings, candidate_embeddings, projection):
    """Return |PCA(q)-PCA(c)|, PCA(q)*PCA(c), and original-embedding cosine distance."""
    query, candidate = _unit_vectors(query_embeddings), _unit_vectors(candidate_embeddings)
    if query.shape != candidate.shape:
        raise ValueError("Pair arrays must have equal shapes")
    q, c = projection.transform(query), projection.transform(candidate)
    cosine_distance = np.clip(1-np.einsum("ij,ij->i", query, candidate), 0, 2)
    return np.c_[np.abs(q-c), q*c, cosine_distance]


def build_pair_tables(embeddings: pd.DataFrame, projection) -> dict[str, pd.DataFrame]:
    """Construct complete train/validation galleries, never test pairs."""
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
