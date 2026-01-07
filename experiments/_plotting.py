from typing import Optional
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
from _analysis import extract_class, load_img, get_embeddings
from PIL import Image
from sklearn.decomposition import PCA

from scipy.stats import gaussian_kde
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import numpy as np
import json
import torchvision.transforms as T

import matplotlib as mpl

mpl.rcParams.update({"text.usetex": True, "font.size": 16,})

COLORS = ["green", "brown", "blue", "grey"]
CMAPS = ["Reds", "Blues"]

def plot_examples(
    orig_paths: list[str], target_paths: list[str], class_mappings: dict[int, str], n_classes:int=10,
    fig_name: str = "default_comp", show_change:bool=False,
    show_bool: bool = False, bool_stats: list[str] | None = None,
    target_classes: list[int] | None = None,
):
    orig_by_class = defaultdict(list)
    stats_by_class = defaultdict(list)
    target_by_class = defaultdict(list)
    tc_by_class = defaultdict(list)


    if target_classes is not None:
        for op, tp, tc in zip(orig_paths, target_paths, target_classes):
            cls = extract_class(op)
            orig_by_class[cls].append(op)
            tc_by_class[cls].append(tc)
            target_by_class[cls].append(tp)
    else:
        for op, tp in zip(orig_paths, target_paths):
            cls = extract_class(op)
            orig_by_class[cls].append(op)
            target_by_class[cls].append(tp)

    if bool_stats is not None:
        for p in bool_stats:
            cls = extract_class(p, mrm=True)
            stats_by_class[cls].append(p)

    classes = sorted(list(orig_by_class.keys()))[:n_classes]

    # Determine number of columns
    n_cols = 3 + int(show_change)
    adder = 0 if target_classes is None else 1
    width_ratios = [(n_cols-1)/n_cols] + [1]*(n_cols-1) + [(n_cols-1)/n_cols]* adder

    fig, axes = plt.subplots(
        n_classes, n_cols + adder,
        figsize=(2.5*(n_cols+ adder ), 3 * n_classes),
        gridspec_kw={'width_ratios': width_ratios}
    )

    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for i, cls in enumerate(classes):
        axes[i, 0].text(
            0.5, 0.5, class_mappings.get(int(cls), cls),
            fontsize=18, fontweight='bold', ha='center', va='center'
        )
        axes[i, 0].axis("off")

        # Load origin image
        o_path = orig_by_class[cls][0]
        o_img = load_img(o_path)
        axes[i, 1].imshow(o_img)
        axes[i, 1].axis("off")

        # Load target image
        t_path = target_by_class[cls][0]
        t_img = load_img(t_path)
        axes[i, 2].imshow(t_img)
        axes[i, 2].axis("off")

        # Extract bool from target filename
        if show_bool:
            if bool_stats is None:
                bool_val = float(t_path.split("_")[-1].replace(".png", ""))
                target_bool_text = str(bool(int(bool_val)))
                origin_bool_text = str(not int(bool_val))
            else:
                with open(stats_by_class[cls][0], "r") as f:
                    data = json.load(f)

                y_0 = data.get("w0_predictions") or data.get("y_0") or data.get("initial_logits")[0]
                y_hat = data.get("best_0_y_hat") or data.get("y_hat") or data.get("final_logits")[0]


                target_bool_text = str(y_hat[int(cls)] > 0)
                origin_bool_text = str(y_0[int(cls)] > 0)


            # Add bool text to corners
            axes[i, 1].text(0.2, 0.9, origin_bool_text, color='black',
                            fontsize=14, ha='right', va='bottom', transform=axes[i, 1].transAxes,
    bbox=dict(facecolor='white', edgecolor='none', pad=2))
            axes[i, 2].text(0.2, 0.9, target_bool_text, color='black',
                            fontsize=14, ha='right', va='bottom', transform=axes[i, 2].transAxes,
    bbox=dict(facecolor='white', edgecolor='none', pad=2))

        # Show change
        if show_change:
            w1, h1, _ = o_img.shape
            w2, h2, _ = t_img.shape

            target_w = min(w1, w2)
            target_h = min(h1, h2)

            o_pil = Image.fromarray(o_img)
            t_pil = Image.fromarray(t_img)

            o_resized = np.array(o_pil.resize((target_w, target_h), Image.BILINEAR), dtype=float)
            t_resized = np.array(t_pil.resize((target_w, target_h), Image.BILINEAR), dtype=float)

            diff = np.abs(np.array(o_resized, dtype=float) - np.array(t_resized, dtype=float)) / 255.0
            axes[i, 3].imshow(diff.mean(axis=-1), cmap="hot")
            axes[i, 3].axis("off")

        # Show predicted class
        if target_classes is not None:
            col_idx = 3 + int(show_change)
            axes[i, col_idx].text(0.5, 0.5, class_mappings.get(tc_by_class[cls][0], tc_by_class[cls][0]),
                                  fontsize=18, fontweight='bold', ha='center', va='center')
            axes[i, col_idx].axis("off")

    # Set titles
    titles = ["Class", "Origin", "Result"] + (["Change"] if show_change else []) \
             + (["Predicted Class"] if target_classes is not None else [])
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontweight='bold', fontsize=18)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.02)
    plt.savefig(f"figures/{fig_name}.pdf", bbox_inches="tight", dpi=200, pad_inches=0)

def plot_embedding_methods(
        methods_embeddings: dict[str, tuple[list[str], list[str]]],
        density: Optional[dict[str, list[str]]] = None,
        fig_name: str="default",
        crop_outliers: Optional[tuple[float, float, float, float]] = None,
):
    """
    Plot origin/target embeddings for multiple methods.

    :param methods_embeddings: dict of {method_name: (origins, targets)}
                               origins/targets are [N, D] torch tensors
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Collect all embeddings for joint PCA projection
    all_embeds = torch.cat([torch.cat([get_embeddings(a), get_embeddings(b)]) for a, b in methods_embeddings.values()])

    pca = PCA(n_components=2)
    reduced = torch.tensor(pca.fit_transform(all_embeds))

    x_all_only, y_all_only = reduced[:, 0].numpy(), reduced[:, 1].numpy()
    x_all, y_all = x_all_only, y_all_only
    if density:
        all_density_pts = torch.cat([get_embeddings(v) for v in density.values()])
        dens_2d = pca.transform(all_density_pts)
        x_all = np.concatenate([x_all, dens_2d[:, 0]])
        y_all = np.concatenate([y_all, dens_2d[:, 1]])

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()

    margin_x = (x_max - x_min) * 0.05
    margin_y = (y_max - y_min) * 0.05
    xi, yi = np.mgrid[x_min-margin_x:x_max+margin_x:200j, y_min-margin_y:y_max+margin_y:200j]

    if density:
        colors = plt.cm.Greys(np.linspace(0, 1, 7))
        colors[0, -1] = 0.0  # set alpha of outermost level to 0
        cmap_custom = mcolors.ListedColormap(colors)

        for label, embeds in density.items():
            emb = get_embeddings(embeds)
            pts_2d = pca.transform(emb)
            x, y = pts_2d[:, 0], pts_2d[:, 1]
            kde = gaussian_kde(np.vstack([x, y]))
            zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=7, cmap=cmap_custom, alpha=0.3)
            ax.contour(xi, yi, zi, levels=7, colors="k", linewidths=0.5, alpha=0.7)

            max_idx = np.unravel_index(np.argmax(zi), zi.shape)
            peak_x, peak_y = xi[max_idx], yi[max_idx]
            ax.text(
                peak_x, peak_y, label,
                ha='center', va='center', color='black', weight='bold'
            )

    offset = 0
    for i, (method, (origins, _)) in enumerate(methods_embeddings.items()):
        n = len(origins)
        reduced_orig = reduced[offset : offset + n]
        reduced_targ = reduced[offset + n : offset + 2 * n]
        offset += 2 * n

        ax.scatter(reduced_orig[:, 0], reduced_orig[:, 1], facecolors='none', edgecolors=COLORS[i], s=60, alpha=0.7)
        ax.scatter(reduced_targ[:, 0], reduced_targ[:, 1], color=COLORS[i], s=60, label=method, alpha=0.7)

    ax.scatter([], [], facecolors='none', edgecolors="grey", s=60, alpha=0.7, label="Origins")
    ax.scatter([], [], facecolors="grey", edgecolors="grey", s=60, alpha=0.7, label="Targets")

    ax.legend(title="Method", loc='lower right')

    if crop_outliers:
        x_lo, x_hi = np.percentile(x_all, crop_outliers[:2])
        y_lo, y_hi = np.percentile(y_all, crop_outliers[2:])

        mx = (x_hi - x_lo) * 0.05
        my = (y_hi - y_lo) * 0.05
        ax.set_xlim(x_lo-mx, x_hi+mx)
        ax.set_ylim(y_lo-my, y_hi+my)

        ax_in = inset_axes(ax, width="28%", height="28%", loc="upper right")

        offset = 0
        for i, (method, (origins, _)) in enumerate(methods_embeddings.items()):
            n = len(origins)
            ax_in.scatter(reduced[offset: offset+2*n, 0], reduced[offset: offset+2*n, 1], color=COLORS[i], s=20, alpha=0.7)
            offset += 2 * n
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        for s in ax_in.spines.values():
            s.set_linewidth(0.5)

        ax_in.plot(
            [x_lo, x_hi, x_hi, x_lo, x_lo],
            [y_lo, y_lo, y_hi, y_hi, y_lo],
            lw=1
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Embedding Projection (Origin vs Target)")
    plt.savefig(f"figures/{fig_name}.pdf", bbox_inches="tight", dpi=200, transparent=True)

def plot_examples_yolo(orig_paths, target_paths, sut, n_classes=10, fig_name="yolo_comp", top_k=5):
    """
    Plot origin and target images with YOLO predictions (bboxes).

    :param orig_paths: list of original image paths
    :param target_paths: list of target image paths
    :param sut: YoloSUT instance
    :param class_mappings: dict mapping class indices to labels
    :param n_classes: how many classes to show
    :param fig_name: filename to save
    """
    orig_by_class = defaultdict(list)
    target_by_class = defaultdict(list)

    for p in orig_paths:
        cls = extract_class(p)
        orig_by_class[cls].append(p)
    for p in target_paths:
        cls = extract_class(p)
        target_by_class[cls].append(p)

    classes = sorted(list(orig_by_class.keys()))[:n_classes]

    fig, axes = plt.subplots(
        n_classes, 3,
        figsize=(10, 3 * n_classes),
        gridspec_kw={'width_ratios': [0.5, 1, 1]}
    )
    axes = np.atleast_2d(axes)

    transform = T.Compose([T.ToTensor()])  # uint8 -> float [0,1]

    for i, cls in enumerate(classes):
        # Class label
        axes[i,0].text(0.5,0.5, sut.class_mapping.get(int(cls), cls),
                        fontsize=18, fontweight='bold', ha='center', va='center')
        axes[i,0].axis("off")

        def process_image(path):
            img = load_img(path)
            tensor = transform(img).unsqueeze(0).to(sut._device)
            preds = sut.process_input(tensor)
            return img, preds

        # Origin image
        o_img, o_preds = process_image(orig_by_class[cls][0])
        axes[i, 1].imshow(draw_yolo_bboxes(o_img, o_preds, sut, top_k, target_class=int(cls)))
        axes[i,1].axis("off")

        # Target image (if exists)
        t_path = target_by_class.get(cls, [None])[0]
        if t_path:
            t_img, t_preds = process_image(t_path)
            axes[i,2].imshow(draw_yolo_bboxes(t_img, t_preds, sut, top_k, target_class=int(cls)))
        axes[i,2].axis("off")

    axes[0,0].set_title("Class", fontweight='bold', fontsize=18)
    axes[0,1].set_title("Origin", fontweight='bold', fontsize=18)
    axes[0,2].set_title("Result", fontweight='bold', fontsize=18)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.02)
    plt.savefig(f"figures/{fig_name}.pdf", bbox_inches="tight", dpi=200, pad_inches=0)

def draw_yolo_bboxes(img, preds, sut, top_k, target_class=None, conf_thresh=0.5):
    """
    Draw YOLO bounding boxes on an image, showing top_k detections
    of target_class with confidence > conf_thresh.
    preds: [1, 84, N] -> first 80: class confidences, last 4: bbox coords
    """
    img_copy = img.copy()
    h_img, w_img, _ = img.shape

    class_conf = preds[0, :80, :]  # [80, N]
    bboxes = preds[0, 80:, :]      # [4, N]

    cls_idx = class_conf.argmax(dim=0)      # [N]
    cls_conf = class_conf.max(dim=0).values # [N]

    # keep only detections of target_class and conf > threshold
    if target_class is not None:
        keep = (cls_idx == target_class) & (cls_conf > conf_thresh)
    else:
        keep = (cls_conf > conf_thresh)

    cls_conf = cls_conf[keep]
    bboxes = bboxes[:, keep]

    if cls_conf.numel() == 0:
        return img_copy

    topk_idx = np.argsort(cls_conf.cpu().numpy())[-top_k:][::-1]

    for i in topk_idx:
        xc, yc, bw, bh = bboxes[:, i].cpu().numpy()
        x1, y1 = int(xc - bw/2), int(yc - bh/2)
        x2, y2 = int(xc + bw/2), int(yc + bh/2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img-1, x2), min(h_img-1, y2)

        conf = cls_conf[i].item()
        cls_name = sut.class_mapping.get(int(target_class), str(target_class))
        label = f"{cls_name} {conf*100:.0f}%"

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_copy

def plot_tradeoff(
        left_metrics: list[tuple[list[np.ndarray], str]],
        right_metrics: list[tuple[list[np.ndarray], str]],
        ticks: list[str],
        colors: list[str],
        name: str,
        n_yticks: int = 6,
        vline: Optional[float] = None,
):
    assert len(left_metrics) + len(right_metrics) <= len(colors), f"Error too little colors ({len(colors)}) defined for {len(left_metrics) + len(right_metrics)} metrics."

    def _get_stats(d):
        means = [v.mean() for v in d]
        lower = [max(0,m-np.percentile(v, 25)) for v,m in zip(d, means)]
        upper = [max(0,np.percentile(v, 75)-m) for v,m in zip(d, means)]
        y_err = [lower, upper]
        return means, y_err

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    if vline is not None:
        ax.axvline(vline, color="r", alpha=0.1, linewidth=12+2*(len(left_metrics)+len(right_metrics)))  # Get middle sweep

    for i, (d, label, bounds) in enumerate(left_metrics):
        a = ax.twinx() if i > 0 else ax
        means, y_err = _get_stats(d)
        color=colors.pop(0)
        a.errorbar([j-(1+i)*0.05 for j in range(len(means))], means, yerr=y_err, fmt="-o", capsize=4, color=color)
        a.set_ylabel(label, color=color)
        a.set_ylim(*bounds)
        a.tick_params(axis="y", colors=color)
        a.spines["left"].set_color(color)
        a.spines["left"].set_position(("axes", 0-(i*0.11)))

        a.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks))


    for i, (d, label, bounds) in enumerate(right_metrics):
        a = ax.twinx()
        means, y_err = _get_stats(d)
        color=colors.pop(0)
        a.errorbar([j+(1+i)*0.05 for j in range(len(means))], means, yerr=y_err, fmt="-o", capsize=4, color=color)
        a.set_ylabel(label, color=color)
        a.set_ylim(*bounds)
        a.tick_params(axis="y", colors=color)
        a.spines["right"].set_color(color)
        a.spines["right"].set_position(("axes", 1+(i*0.11)))

        a.yaxis.set_major_locator(MaxNLocator(nbins=n_yticks))


    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right")

    ax.set_xlabel("Learning Rate Schedule")
    ax.set_xlim([-0.2,len(ticks)-0.8])

    ax.grid(axis="y", color="gray", alpha=0.3)
    plt.savefig(f"figures/sweep_{name}.pdf", dpi=200, bbox_inches="tight")
