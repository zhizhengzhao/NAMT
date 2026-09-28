from __future__ import annotations

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

METHODS = [
    ("PoCA", "poca", "#777777", "o", "-"),
    ("ASR", "asr_q75", "#DD8A27", "^", "-."),
    ("MLSD", "mlsd_median", "#3B73B9", "s", "--"),
    ("NAMT-3P", "namt_3p", "#2E8B57", "D", "-"),
    ("NAMT-4P", "namt_4p", "#B23A48", "P", "-"),
]


SCENES = [
    ("lead", "concrete", "u_pb_concrete"),
    ("lead", "soil", "u_pb_soil"),
    ("RDX", "concrete", "u_explosive_concrete"),
    ("RDX", "soil", "u_explosive_soil"),
    ("air void", "concrete", "u_void_concrete"),
    ("air void", "soil", "u_void_soil"),
]


CONDITIONS = [
    ("150k/1 mm", "150k_1mm"),
    ("150k/2 mm", "150k_2mm"),
    ("150k/3 mm", "150k_3mm"),
    ("100k/1 mm", "100k_1mm"),
    ("50k/1 mm", "50k_1mm"),
]


TABLE_CONDITION_LABELS = {
    "150k_2mm": "150,000 / 2 mm",
    "150k_3mm": "150,000 / 3 mm",
    "100k_1mm": "100,000 / 1 mm",
    "50k_1mm": "50,000 / 1 mm",
}


REFERENCE_CONDITION = "150k_1mm"


GALLERY_SEED = 43


GROUP_COLORS = ["#EFD8CF", "#D6E2EE", "#DCE8D5"]


BG_COLORS = {"concrete": "#EEE7DD", "soil": "#E0E5EB"}


def format_stat(mean: float, std: float, best: float, digits: int = 3) -> str:
    threshold = 0.5 * 10 ** (-digits)
    if abs(mean) < threshold:
        mean = 0.0
    if abs(std) < threshold:
        std = 0.0
    rendered_mean = f"{mean:.{digits}f}"
    rendered_std = f"{std:.{digits}f}"


    if rendered_mean == f"{best:.{digits}f}":
        rendered_mean = rf"\best{{{rendered_mean}}}"
    return rf"{rendered_mean}\uncert{{{rendered_std}}}"


def macro_by_seed(
    runs: dict[tuple[str, int, str, str], dict[str, float]],
    seeds: tuple[int, ...],
    condition: str,
    method: str,
    metric: str,
    scenes: list[str],
) -> np.ndarray:
    return np.asarray(
        [np.mean([runs[(condition, seed, method, scene)][metric] for scene in scenes]) for seed in seeds],
        dtype=np.float64,
    )


def make_robustness_plot(
    output: Path, runs: dict[tuple[str, int, str, str], dict[str, float]], seeds: tuple[int, ...]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.45), sharey=False)
    panels = [
        ([1, 2, 3], ["150k_1mm", "150k_2mm", "150k_3mm"], "hit error (mm)"),
        ([50, 100, 150], ["50k_1mm", "100k_1mm", "150k_1mm"], r"accepted events ($\times 10^3$)"),
    ]
    metrics = [
        ("auc", "macro-average AUC", (0.48, 1.00)),
        ("cnr", "macro-average CNR", (0.0, 7.5)),
    ]
    titles = [
        ["(a) AUC (150,000 accepted events)", "(b) AUC (1 mm hit error)"],
        ["(c) CNR (150,000 accepted events)", "(d) CNR (1 mm hit error)"],
    ]
    all_scenes = [scene for *_, scene in SCENES]
    for row, (metric, ylabel, ylim) in enumerate(metrics):
        for col, (x, conditions, xlabel) in enumerate(panels):
            ax = axes[row, col]
            for label, method, color, marker, linestyle in METHODS:
                statistics = [
                    macro_by_seed(runs, seeds, condition, method, metric, all_scenes)
                    for condition in conditions
                ]
                y = [values.mean() for values in statistics]
                yerr = [values.std(ddof=1) for values in statistics]
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.55,
                    markersize=4.2,
                    markeredgewidth=0.7,
                    elinewidth=0.75,
                    capsize=1.8,
                    capthick=0.75,
                )
            if metric == "auc":
                ax.axhline(0.5, color="#9A9A9A", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_title(titles[row][col], fontsize=8.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_ylim(*ylim)
            ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.105, top=0.89, wspace=0.27, hspace=0.48)
    fig.savefig(output / "fig4_robustness.pdf")
    plt.close(fig)


def normalized_panel(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        image = np.asarray(data["recon"], dtype=np.float64)
        background = np.asarray(data["bg_roi"], dtype=bool) & np.isfinite(image)
        gx = np.asarray(data["gx"], dtype=np.float64)
        gy = np.asarray(data["gy"], dtype=np.float64)
    mean = float(image[background].mean())
    sd = float(image[background].std())
    if not np.isfinite(sd) or sd == 0.0:
        sd = 1.0
    return (image - mean) / sd, gx, gy


def make_gallery(repo: Path, output: Path, seed: int) -> None:
    n_rows, n_cols = len(METHODS), len(SCENES)


    figure_width, figure_height = 5.72, 3.85
    left, right = 0.110, 0.995
    bottom = 0.060 / figure_height
    row_region = 3.203 / figure_height
    top = bottom + row_region
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_width, figure_height))
    for row, (_, method, *_rest) in enumerate(METHODS):
        for col, (_, _, scene) in enumerate(SCENES):
            path = repo / REFERENCE_CONDITION / f"seed_{seed}" / method / f"{scene}.npz"
            z, gx, gy = normalized_panel(path)
            ax = axes[row, col]


            ax.imshow(
                np.flipud(np.clip(z, -4.0, 4.0)),
                origin="lower",
                extent=[gx[0], gx[-1], gy[0], gy[-1]],
                cmap="RdBu_r",
                vmin=-4.0,
                vmax=4.0,
                interpolation="nearest",
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.015, hspace=0.018)

    col_width = (right - left) / n_cols
    row_height = (top - bottom) / n_rows
    gap = 0.004
    regular_font = font_manager.FontProperties(family="DejaVu Sans")

    def block(
        x0: float,
        width: float,
        y_top: float,
        height: float,
        color: str,
        label: str,
        size: float,
    ) -> None:
        fig.add_artist(
            Rectangle(
                (x0 + gap / 2, y_top - height + gap / 2),
                width - gap,
                height - gap,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            )
        )
        if label:
            fig.text(
                x0 + width / 2,
                y_top - height / 2,
                label,
                ha="center",
                va="center",
                fontsize=size,
                fontproperties=regular_font,
            )

    method_header_height = 0.130 / figure_height
    background_height = 0.160 / figure_height
    material_height = 0.240 / figure_height
    for col, (_, background, _) in enumerate(SCENES):
        block(
            left + col * col_width,
            col_width,
            top + method_header_height + background_height,
            background_height,
            BG_COLORS[background],
            background,
            6.7,
        )
    for group, (material, _, _) in enumerate(SCENES[::2]):
        block(
            left + 2 * group * col_width,
            2 * col_width,
            top + method_header_height + background_height + material_height,
            material_height,
            GROUP_COLORS[group],
            material,
            7.6,
        )
    block(
        0.0,
        left,
        top + method_header_height + background_height + material_height,
        material_height,
        "#B9B9B9",
        "object\nmaterial",
        6.4,
    )
    block(
        0.0,
        left,
        top + method_header_height + background_height,
        background_height,
        "#D2D2D2",
        "background",
        6.2,
    )
    block(0.0, left, top + method_header_height, method_header_height, "#E0E0E0", "method", 6.4)
    fig.add_artist(
        Line2D(
            [left + gap / 2, right - gap / 2],
            [top + gap / 2, top + gap / 2],
            transform=fig.transFigure,
            color="#D2D2D2",
            linewidth=0.55,
            clip_on=False,
        )
    )
    for row, (label, *_rest) in enumerate(METHODS):
        block(
            0.0,
            left,
            top - row * row_height,
            row_height,
            "#F3F3F3",
            label,
            7.2,
        )
    divider_y = top - 3 * row_height
    fig.add_artist(
        Line2D(
            [gap / 2, right - gap / 2],
            [divider_y, divider_y],
            transform=fig.transFigure,
            color="#B8B8B8",
            linewidth=0.7,
            clip_on=False,
        )
    )
    fig.savefig(output / "fig3_reference_gallery.pdf")
    plt.close(fig)


def table_rows(summaries, condition, metric):
    digits = 3 if metric == "auc" else 2
    ordered_scenes = [scene for _, _, scene in SCENES]
    best = [max(summaries[(condition, method, scene)][f"{metric}_mean"] for _, method, *_ in METHODS) for scene in ordered_scenes]
    lines = []
    for index, (label, method, *_) in enumerate(METHODS):
        cells = [
            format_stat(
                summaries[(condition, method, scene)][f"{metric}_mean"],
                summaries[(condition, method, scene)][f"{metric}_std"], best[column], digits,
            )
            for column, scene in enumerate(ordered_scenes)
        ]
        lines.append((index, label, cells))
    return lines


def write_benchmark_tables(summaries, output):
    lines = []
    for metric in ("auc", "cnr"):
        if lines:
            lines.append(r"\midrule")
        for index, label, cells in table_rows(summaries, REFERENCE_CONDITION, metric):
            prefix = rf"\multirow{{5}}{{*}}{{{metric.upper()}}}" if index == 0 else ""
            lines.append(prefix + " & " + label + " & " + " & ".join(cells) + r"\\")
    (output / "table2_reference.tex").write_text("\n".join(lines) + "\n")
    for metric, number in (("auc", 4), ("cnr", 5)):
        lines = []
        for condition, label in TABLE_CONDITION_LABELS.items():
            if lines:
                lines.append(r"\addlinespace[2pt]")
            for index, method, cells in table_rows(summaries, condition, metric):
                prefix = rf"\multirow{{5}}{{*}}{{{label}}}" if index == 0 else ""
                lines.append(prefix + " & " + method + " & " + " & ".join(cells) + r"\\")
        (output / f"table{number}_{metric}.tex").write_text("\n".join(lines) + "\n")


def ablation_report(root, output):
    from ablation.run import SCENES as scenes, VARIANTS
    from evaluation.artifact import load_artifact
    from evaluation.metrics import score_artifact
    from evaluation.evaluate import write_csv
    from namt.config import SEEDS

    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            for scene in scenes:
                path = root / f"seed_{seed}" / variant / f"{scene}.npz"
                with np.load(path, allow_pickle=False) as data:
                    if float(data["sigma0"]) != 1.0 or float(data["slope"]) != 0.15:
                        raise ValueError(f"incorrect ablation detector: {path}")
                    if int(data["seed"]) != seed or str(data["variant"]) != variant or str(data["scene"]) != scene:
                        raise ValueError(f"incorrect ablation metadata: {path}")
                rows.append({"seed": seed, "variant": variant, "scene": scene, **score_artifact(load_artifact(path, scene, 120.0))})
    write_csv(output / "ablation_metrics.csv", rows, tuple(rows[0]))
    summary = []
    for variant in VARIANTS:
        for scene in scenes:
            selected = [row for row in rows if row["variant"] == variant and row["scene"] == scene]
            item = {"variant": variant, "scene": scene, "n": len(selected)}
            for metric in ("auc", "cnr"):
                values = np.array([row[metric] for row in selected])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=1))
            summary.append(item)
    write_csv(output / "ablation_summary.csv", summary, tuple(summary[0]))
    labels = {
        "full": "full", "gaisser_prior": "Gaisser spectrum", "fixed_momentum": "fixed momentum",
        "analytic_instrument": "nominal instrument", "no_instrument": "no instrument",
    }
    lines = []
    for metric, digits in (("auc", 3), ("cnr", 2)):
        if lines:
            lines.append(r"\midrule")
        for index, variant in enumerate(VARIANTS):
            prefix = rf"\multirow{{5}}{{*}}{{{metric.upper()}}}" if index == 0 else ""
            cells = []
            for scene in scenes:
                item = next(item for item in summary if item["variant"] == variant and item["scene"] == scene)
                cells.append(f"{item[f'{metric}_mean']:.{digits}f}" + r"\uncert{" + f"{item[f'{metric}_std']:.{digits}f}" + "}")
            lines.append(prefix + " & " + labels[variant] + " & " + " & ".join(cells) + r"\\")
    (output / "table3_ablation.tex").write_text("\n".join(lines) + "\n")


def build_reports(root, output, suite="all"):
    from evaluation.evaluate import reconstruction_matrix, validate_matrix, evaluate, summarize, write_csv, RUN_FIELDS
    from namt.config import SEEDS

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    if suite in ("all", "benchmark"):
        benchmark = root / "benchmark"
        matrix = reconstruction_matrix(benchmark, SEEDS)
        validate_matrix(benchmark, matrix)
        rows = evaluate(matrix, benchmark, 120.0)
        summary = summarize(rows)
        write_csv(output / "benchmark_metrics.csv", rows, RUN_FIELDS)
        write_csv(output / "benchmark_summary.csv", summary, tuple(summary[0]))
        summaries = {(row["condition"], row["method"], row["scene"]): row for row in summary}
        runs = {(row["condition"], row["seed"], row["method"], row["scene"]): row for row in rows}
        write_benchmark_tables(summaries, output)
        make_gallery(benchmark, output, GALLERY_SEED)
        make_robustness_plot(output, runs, SEEDS)
        macro = []
        for _, condition in CONDITIONS:
            for _, method, *_ in METHODS:
                item = {"condition": condition, "method": method}
                for metric in ("auc", "cnr"):
                    values = macro_by_seed(runs, SEEDS, condition, method, metric, [scene for _, _, scene in SCENES])
                    item[f"{metric}_mean"] = float(values.mean())
                    item[f"{metric}_std"] = float(values.std(ddof=1))
                macro.append(item)
        write_csv(output / "robustness_summary.csv", macro, tuple(macro[0]))
    if suite in ("all", "ablation"):
        ablation_report(root / "ablation", output)
    print(f"saved reports to {output}", flush=True)
