import pandas as pd
import json
import lpips
from PIL import Image
from torchvision import models, transforms
import torch
import tempfile
import shutil
from pathlib import Path
import numpy as np
from collections import defaultdict
from sewar import msssim
import re
from glob import glob
from pytorch_fid.fid_score import InceptionV3, compute_statistics_of_path, calculate_frechet_distance

lpips_model = lpips.LPIPS(net='vgg').eval()
_real_stats_cache = {}

def get_origins_targets(path: str, origin_patter: str, target_patter: str) -> tuple[list[str], list[str]]:
    return glob(path + origin_patter), glob(path + target_patter)

def load_img(path: str, return_img: bool = False):
    img = np.load(path) if path.endswith(".npy") else np.array(Image.open(path).convert("RGB"))
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (
                img.max() - img.min()
        )
    if img.ndim == 3 and img.shape[0] in [1,3]:  # (C,H,W) -> (H,W,C)
            img = np.transpose(img, (1,2,0))
    if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).astype(np.uint8)
    return Image.fromarray(img) if return_img else img

def get_embedding_diversity(embeddings: torch.Tensor):
    return embeddings.var(dim=0).mean().item()

def get_trace_difference(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor):
    return (torch.diag(torch.cov(embeddings_a.T)) - torch.diag(torch.cov(embeddings_b.T))).abs().sum().item()

def get_embeddings(paths: list[str], batch_size: int = 32):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    feats = []

    for i in range(0, len(paths), batch_size):
        batch = [tfm(load_img(p, return_img=True)) for p in paths[i:i+batch_size]]

        x = torch.stack(batch)
        with torch.no_grad():
            f = model(x).cpu()
        feats.append(f)
    embeddings = torch.cat(feats)
    return embeddings

def get_ssim(origin_paths: list[str], target_paths: list[str]) -> list[float]:
    scores = []
    for origin, target in zip(origin_paths, target_paths):
        im1 = load_img(origin)
        im2 = load_img(target)
        if im1.shape != im2.shape:
            im2 = np.array(Image.fromarray(im2).resize((im1.shape[1], im1.shape[0])))

        score = msssim(im1, im2).real
        scores.append(score)
    return scores


def _to_lpips_tensor(arr):
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = (t * 2) - 1
    return t

def get_lpips(origin_paths: list[str], target_paths: list[str]) -> list[float]:
    scores = []
    for origin, target in zip(origin_paths, target_paths):
        im1 = load_img(origin)
        im2 = load_img(target)

        if im1.shape != im2.shape:
            im2 = np.array(
                Image.fromarray(im2).resize((im1.shape[1], im1.shape[0]), Image.BICUBIC)
            )
        t1 = _to_lpips_tensor(im1)
        t2 = _to_lpips_tensor(im2)

        with torch.no_grad():
            d = lpips_model(t1, t2).item()

        scores.append(d)
    return scores

def load_jsons(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        rows.append({"file": p, **data})
    return pd.DataFrame(rows)

def get_class_stats(orig_preds, target_preds):
    orig = np.array(orig_preds)  # [N, C]
    target = np.array(target_preds)

    top2 = np.argsort(orig, axis=1)[:, -2:]
    top1_orig = top2[:, 1]
    top2_orig = top2[:, 0]


    top1_target = np.argmax(target, axis=1)
    not_top1 = (top1_target != top1_orig).mean()
    as_top2 = (top1_target == top2_orig).mean()
    return not_top1, 1-as_top2

def get_class_flip_stats(orig_preds, target_preds, target_indices):
    orig = np.array(orig_preds)
    target = np.array(target_preds)
    flips = np.sign(orig) != np.sign(target)

    target_flips = [flips[i, int(idx)] for i, idx in enumerate(target_indices)]
    target_flip_fraction = np.mean(target_flips)

    non_target_flips = []
    for i, idx in enumerate(target_indices):
        f = np.delete(flips[i], int(idx))
        non_target_flips.append(f.mean())
    sensitivity = np.mean(non_target_flips)

    return target_flip_fraction, sensitivity

def get_yolo_class_stats(orig_preds, target_preds, target_indices, top_k=5):
    """
    Compute per-class fractional decrease, confidence decrease, and initial confidence.
    Returns dict[class] = {'frac': [...], 'conf': [...], 'init_conf': [...]}
    """
    orig_preds = np.array(orig_preds)
    orig_preds = orig_preds.transpose(0,2,1)
    target_preds = np.array(target_preds)
    if target_preds.shape != orig_preds.shape:
        target_preds = target_preds.transpose(0,2,1)
    N, D, C = orig_preds.shape

    class_stats = defaultdict(lambda: {'frac': [], 'conf_decrease': []})

    for i in range(N):
        sorted_idx_o = np.argsort(orig_preds[i].max(axis=1))[::-1]
        sorted_idx_t = np.argsort(target_preds[i].max(axis=1))[::-1]
        sorted_origin = orig_preds[i, sorted_idx_o[:top_k], :]  # [top_k, C]
        sorted_target = target_preds[i, sorted_idx_t[:top_k], :]  # [top_k, C]
        ti = int(target_indices[i])

        # Fractional decrease
        orig_count = np.sum(sorted_origin.argmax(axis=-1) == ti)
        targ_count = np.sum(sorted_target.argmax(axis=-1) == ti)
        frac_decrease = (orig_count - targ_count) / orig_count if orig_count > 0 else 0
        class_stats[ti]['frac'].append(frac_decrease)

        # Confidence decrease
        idx_target = sorted_origin.argmax(axis=-1) == ti

        if idx_target.sum() > 0:
            orig_conf = sorted_origin[idx_target, ti].mean()
            targ_conf = sorted_target[idx_target, ti].mean()

            conf_dec = 1 - (targ_conf / orig_conf) if orig_conf > 0 else 0.0
        else:
            conf_dec = 0.0

        class_stats[ti]['conf_decrease'].append(conf_dec)
    return class_stats


def print_im_div(*pairs, tools, experiment):
    print(f"{experiment} Image Diversity:")
    for pair, tool in zip(pairs, tools):
        if not pair[0] or not pair[1]:
            print(f"\t{tool}: no data (empty path list)")
            continue
        origin, final = get_embeddings(pair[0], batch_size=1), get_embeddings(pair[1], batch_size=1)
        print(f"\t{tool} Origin Diversity: {get_embedding_diversity(origin):.3f}")
        print(f"\t{tool} Target Diversity: {get_embedding_diversity(final):.3f}")
        print(f"\t{tool} Trace diff: {get_trace_difference(origin, final):.3f}")

def extract_class(path, mrm=False):
    match = re.search(r'class_(\d+)', path)
    if match:
        return match.group(1)
    match = re.search(r'_X(\d+)', path)
    if match:
        return match.group(1)
    return path.split("_")[-1].split(".")[0] if mrm else None

def align_mrm_lists(origins: list[str], targets: list[str]) -> tuple[list[str], list[str]]:
    """Aligns and sorts origins and targets by 'image_<A>_X<B>' key."""
    origin_map = defaultdict(list)
    target_map = defaultdict(list)

    # Build multi-entry maps
    for p in origins:
        key = extract_class(p)
        if key:
            origin_map[key].append(p)

    for p in targets:
        key = extract_class(p)
        if key:
            target_map[key].append(p)

    # Get shared keys (only ones with both origin and target)
    shared_keys = sorted(set(origin_map) & set(target_map))

    aligned_origins, aligned_targets = [], []
    for key in shared_keys:
        o_list, t_list = sorted(origin_map[key]), sorted(target_map[key])
        n = min(len(o_list), len(t_list))
        aligned_origins.extend(o_list[:n])
        aligned_targets.extend(t_list[:n])

    return aligned_origins, aligned_targets

def get_fid(real_paths: list[str], fake_paths: list[str], device: str = None, batch_size:int=128) -> float:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with tempfile.TemporaryDirectory() as temp_dir:
        real_dir = Path(temp_dir) / "real"
        fake_dir = Path(temp_dir) / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()

        for i, path in enumerate(real_paths):
            ext = ".jpg" if (ext_t := Path(path).suffix.lower()) == ".jpeg" else ext_t
            shutil.copy2(path, real_dir / f"real_{i}{ext}")

        for i, path in enumerate(fake_paths):
            ext = Path(path).suffix.lower()
            shutil.copy2(path, fake_dir / f"fake_{i}{ext}")

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)

        real_key = tuple(real_paths)
        if real_key in _real_stats_cache:
            m1, s1 = _real_stats_cache[real_key]
        else:
            m1, s1 = compute_statistics_of_path(str(real_dir), model, batch_size, 2048, device, 1)
            _real_stats_cache[real_key] = (m1, s1)

        m2, s2 = compute_statistics_of_path(str(fake_dir), model, batch_size,2048, device, 1)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value