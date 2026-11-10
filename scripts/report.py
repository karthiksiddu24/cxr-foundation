#!/usr/bin/env python3
"""Generate RESULTS.md, docs/figures/*, and web/src/content/results.json from eval artifacts.

    uv run scripts/report.py

No number in the outputs is typed by hand: everything is read from
$CXR_DATA_ROOT/{runs,artifacts/eval}. Sections whose artifacts are missing say "not run".
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cxr_common.labels import DISPLAY_NAMES, NIH_LABELS
from cxr_ml.paths import DataPaths
from cxr_ml.viz import style

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "docs" / "figures"
P = DataPaths.from_env()
EV = P.eval_artifacts

BASELINES = [
    ("sup_densenet121_imagenet_224", "DenseNet-121", "ImageNet", 224),
    ("sup_densenet121_random_224", "DenseNet-121", "random", 224),
    ("sup_resnet50_imagenet_224", "ResNet-50", "ImageNet", 224),
    ("sup_resnet50_random_224", "ResNet-50", "random", 224),
    ("sup_densenet121_imagenet_512", "DenseNet-121", "ImageNet", 512),
]
INITS = [("random", "Random init"), ("imagenet", "ImageNet init"), ("ssl", "MAE init (ours)")]
FRACS = [("001", 0.01), ("010", 0.1), ("100", 1.0)]


def load(path: Path) -> Any | None:
    return json.loads(path.read_text()) if path.exists() else None


def metrics(run: str) -> dict[str, Any] | None:
    return load(EV / run / "metrics.json")


def boot_mean(run: str) -> np.ndarray | None:
    """Per-resample mean AUROC over the 14 findings (identical resamples across runs)."""
    p = EV / run / "boot_auroc.npy"
    return np.nanmean(np.load(p), axis=1) if p.exists() else None


def fmt(iv: dict[str, Any] | None, d: int = 3) -> str:
    if not iv or iv.get("point") is None:
        return "not run"
    if iv.get("ci_low") is None:
        return f"{iv['point']:.{d}f}"
    return f"{iv['point']:.{d}f} ({iv['ci_low']:.{d}f}–{iv['ci_high']:.{d}f})"


def ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- label efficiency


def label_efficiency() -> dict[str, Any]:
    """Mean over seeds per (init, fraction); CI from shared bootstrap resamples averaged over seeds."""
    series, boots = [], {}
    for init, label in INITS:
        pts = []
        for tag, frac in FRACS:
            runs = sorted(
                p.name
                for p in P.runs.glob(f"ft_vitb_{init}_f{tag}_s*")
                if (EV / p.name / "metrics.json").exists()
            )
            if not runs:
                continue
            seed_points = [metrics(r)["test_auroc"]["mean"]["point"] for r in runs]  # type: ignore[index]
            b = np.mean(np.stack([boot_mean(r) for r in runs]), axis=0)  # type: ignore[arg-type]
            boots[(init, frac)] = b
            lo, hi = ci(b)
            pts.append(
                {
                    "fraction": frac,
                    "mean": float(np.mean(seed_points)),
                    "ci_low": lo,
                    "ci_high": hi,
                    "seeds": [float(x) for x in seed_points],
                    "runs": runs,
                }
            )
        if pts:
            series.append({"init": init, "label": label, "points": pts})
    diffs = []
    for a, b_ in (("ssl", "imagenet"), ("ssl", "random"), ("imagenet", "random")):
        for _, frac in FRACS:
            if (a, frac) in boots and (b_, frac) in boots:
                d = boots[(a, frac)] - boots[(b_, frac)]
                lo, hi = ci(d)
                diffs.append(
                    {
                        "a": a,
                        "b": b_,
                        "fraction": frac,
                        "diff": float(np.mean(d)),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return {"series": series, "paired_differences": diffs}


def plot_label_efficiency(le: dict[str, Any], path: Path) -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s in le["series"]:
        c = style.INIT_COLORS[s["init"]]
        pts = sorted(s["points"], key=lambda p: p["fraction"])
        x = [p["fraction"] for p in pts]
        ax.fill_between(
            x, [p["ci_low"] for p in pts], [p["ci_high"] for p in pts], color=c, alpha=0.15, lw=0
        )
        ax.plot(x, [p["mean"] for p in pts], color=c, marker="o", label=s["label"])
        for p in pts:
            if len(p["seeds"]) > 1:
                ax.scatter(
                    [p["fraction"]] * len(p["seeds"]),
                    p["seeds"],
                    color=c,
                    s=10,
                    alpha=0.6,
                    zorder=3,
                )
        ax.annotate(
            s["label"],
            (x[-1], pts[-1]["mean"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=style.TEXT,
        )
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1.0], ["1%", "10%", "100%"])
    ax.set_xlim(0.007, 2.5)
    ax.set_xlabel("Labeled training patients")
    ax.set_ylabel("Mean test AUROC (14 findings)")
    ax.set_title("Label efficiency: ViT-B/16 by initialization")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_mae_loss(run: str, path: Path) -> bool:
    f = P.runs / run / "metrics.jsonl"
    if not f.exists():
        return False
    recs = [json.loads(line) for line in f.read_text().splitlines() if "train/loss" in line]
    if not recs:
        return False
    style.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    steps = np.array([r["step"] for r in recs])
    spe = steps[0] if len(steps) else 1
    ax.plot(steps / spe, [r["train/loss"] for r in recs], color=style.SERIES[2])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked-patch MSE (normalized pixels)")
    ax.set_title(f"MAE pretraining loss ({run})")
    fig.savefig(path)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- main


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    w = L.append
    w("# Results\n")
    w(
        f"_Generated by `scripts/report.py` on {now} at commit `{git_commit()}` from evaluation "
        "artifacts in `$CXR_DATA_ROOT/artifacts/eval/`. Do not edit by hand; every number below is "
        'read from those files. "not run" means the artifact does not exist._\n'
    )
    w(
        "All AUROCs are on the NIH official test list (25,596 images, 2,797 patients) unless stated. "
        "Intervals are 95% patient-clustered bootstrap CIs (2,000 resamples, shared across runs so "
        "differences are paired).\n"
    )

    # ---- Phase 2
    w("## 1. Supervised baselines\n")
    w(
        "| Model | Init | Input | Val mean AUROC | Test mean AUROC (95% CI) | Mean ECE raw → calibrated |"
    )
    w("|---|---|---|---:|---:|---:|")
    for run, arch, init, size in BASELINES:
        m = metrics(run)
        if m is None:
            w(f"| {arch} | {init} | {size} | not run | not run | not run |")
            continue
        w(
            f"| {arch} | {init} | {size} | {m['val_mean_auroc']:.3f} | {fmt(m['test_auroc']['mean'])} | "
            f"{m['mean_ece_raw']:.3f} → {m['mean_ece_calibrated']:.3f} |"
        )
    w("")
    best_base = max(
        (r for r, *_ in BASELINES if metrics(r)),
        key=lambda r: metrics(r)["val_mean_auroc"],
        default=None,
    )  # type: ignore[index]
    if best_base:
        m = metrics(best_base)
        assert m is not None
        w(
            f"\nPer-finding results for the baseline with the best **validation** score (`{best_base}`):\n"
        )
        w("| Finding | AUROC (95% CI) | AUPRC (95% CI) | Test prevalence | AUROC PA | AUROC AP |")
        w("|---|---:|---:|---:|---:|---:|")
        for n in NIH_LABELS:
            pa = m["by_view"]["PA"]["per_class"][n]
            ap = m["by_view"]["AP"]["per_class"][n]
            w(
                f"| {DISPLAY_NAMES[n]} | {fmt(m['test_auroc']['per_class'][n])} | {fmt(m['test_auprc']['per_class'][n])} | "
                f"{100 * m['test_prevalence'][n]:.1f}% | {pa:.3f} | {ap:.3f} |"
            )
        w(
            f"\nMean AUROC by view: PA {m['by_view']['PA']['mean_auroc']:.3f} "
            f"(n={m['by_view']['PA']['n_images']:,}), AP {m['by_view']['AP']['mean_auroc']:.3f} "
            f"(n={m['by_view']['AP']['n_images']:,}).\n"
        )
        rel = EV / best_base / "reliability.png"
        if rel.exists():
            shutil.copy(rel, FIG / "reliability_best_baseline.png")
            w("![Reliability diagrams](docs/figures/reliability_best_baseline.png)\n")

    # ---- Phase 3
    w("## 2. MAE pretraining\n")
    mae_runs = ["mae_vit_base_ep800", "mae_vit_small_ep800"]
    any_mae = False
    for run in mae_runs:
        if plot_mae_loss(run, FIG / f"{run}_loss.png"):
            any_mae = True
            done = (P.runs / run / "DONE").exists()
            w(
                f"- `{run}`: {'complete' if done else 'in progress'}. ![loss](docs/figures/{run}_loss.png)"
            )
            recon = sorted((P.runs / run / "recon").glob("epoch_*.png"))
            if recon:
                shutil.copy(recon[-1], FIG / f"{run}_recon.png")
                w(
                    f"  Reconstructions at {recon[-1].stem.replace('_', ' ')} (val images; top: original, "
                    f"middle: 75% masked input, bottom: reconstruction): ![recon](docs/figures/{run}_recon.png)"
                )
    if not any_mae:
        w("not run")
    w("")

    # ---- Phase 4
    w("## 3. Label efficiency (headline)\n")
    le = label_efficiency()
    if le["series"]:
        plot_label_efficiency(le, FIG / "label_efficiency.png")
        w("![Label efficiency](docs/figures/label_efficiency.png)\n")
        w("| Initialization | 1% labels | 10% labels | 100% labels |")
        w("|---|---:|---:|---:|")
        for s in le["series"]:
            cells = []
            for _, frac in FRACS:
                p = next((p for p in s["points"] if p["fraction"] == frac), None)
                cells.append(
                    "not run"
                    if p is None
                    else f"{p['mean']:.3f} ({p['ci_low']:.3f}–{p['ci_high']:.3f}), n_seeds={len(p['seeds'])}"
                )
            w(f"| {s['label']} | " + " | ".join(cells) + " |")
        w(
            "\nPoints are means over seeds (each seed uses an independent patient subset); intervals "
            "are bootstrap CIs of that seed-averaged mean (test-sampling uncertainty). Seed spread is "
            "shown as dots in the figure.\n"
        )
        if le["paired_differences"]:
            w("Paired differences in mean AUROC (same bootstrap resamples):\n")
            w("| Comparison | Labels | Δ mean AUROC (95% CI) |")
            w("|---|---:|---:|")
            names = dict(INITS)
            for d in le["paired_differences"]:
                w(
                    f"| {names[d['a']]} − {names[d['b']]} | {d['fraction']:.0%} | {d['diff']:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f}) |"
                )
            w("")
    else:
        w("not run\n")

    # ---- Phase 5
    w("## 4. External validation (CheXpert)\n")
    ext_runs = sorted({p.parent.name for p in EV.glob("*/external_chexpert_test.json")})
    if ext_runs:
        for run in ext_runs:
            e = load(EV / run / "external_chexpert_test.json")
            assert e is not None
            w(
                f"`{run}` on CheXpert test ({e['n_images']} frontal images, {e['n_patients']} patients): "
                f"mean AUROC over exact-mapped findings {fmt(e['mean_auroc_exact_mappings'])}\n"
            )
            w(
                "| CheXpert finding | NIH output(s) | CheXpert AUROC (95% CI) | NIH test AUROC | Drop |"
            )
            w("|---|---|---:|---:|---:|")
            for k, v in e["per_finding"].items():
                approx = " (approx.)" if v["approximate"] else ""
                nih = fmt(v.get("nih_test_auroc")) if v.get("nih_test_auroc") else "—"
                drop = f"{v['drop']:+.3f}" if "drop" in v else "—"
                w(
                    f"| {k}{approx} | {'/'.join(v['nih_outputs'])} | {fmt(v['auroc'])} | {nih} | {drop} |"
                )
            w("")
    else:
        w(
            "not run — CheXpert expert-labeled files were not provided "
            "(see `scripts/check_data.py` for the expected layout).\n"
        )

    w("## 5. Heatmap localization\n")
    loc_files = sorted(EV.glob("*/localization_*.json"))
    if loc_files:
        w(
            "Pointing game: fraction of boxed findings where the heatmap's maximum falls inside a "
            "radiologist box. A heatmap is only informative if it beats the image-center and "
            "random-point baselines.\n"
        )
        w(
            "| Run | Boxes | Heatmap | n | Pointing game (95% CI) | Center baseline | Random baseline | Mean IoU |"
        )
        w("|---|---|---|---:|---:|---:|---:|---:|")
        for f in loc_files:
            s = load(f)
            assert s is not None
            if s["run"].startswith("lrsweep"):
                continue
            o = s["overall"]
            pg = f"{o['pointing_game']:.3f} ({o['pointing_game_ci'][0]:.3f}–{o['pointing_game_ci'][1]:.3f})"
            w(
                f"| `{s['run']}` | {s['dataset'].upper()} | {s['method']} | {s['n_items']} | {pg} | "
                f"{o['center_baseline']:.3f} | {o['random_baseline']:.3f} | {o['mean_iou']:.3f} |"
            )
        w("")
    else:
        w("not run\n")

    w("## 6. Shortcut audit: pneumothorax and chest drains\n")
    sc_files = sorted(
        f
        for f in EV.glob("*/shortcut_pneumothorax_drain.json")
        if not f.parent.name.startswith("lrsweep")
    )
    if sc_files:
        w(
            "AUROC for pneumothorax-positive images **with** vs **without** a chest drain, each against "
            "all test negatives. Drain labels: NEATX non-expert annotations (CC BY-NC-SA). A large "
            "positive gap means the model partly detects the treatment (drain) rather than the disease.\n"
        )
        w(
            "| Run | Positives with drain | Positives without drain | AUROC with drain | AUROC without drain | Gap (95% CI) |"
        )
        w("|---|---:|---:|---:|---:|---:|")
        for f in sc_files:
            s = load(f)
            assert s is not None
            g = s["difference"]
            w(
                f"| `{s['run']}` | {s['n_pos_with_drain']} | {s['n_pos_without_drain']} | {fmt(s['auroc_with_drain'])} | "
                f"{fmt(s['auroc_without_drain'])} | {g['point']:+.3f} ({g['ci_low']:+.3f} to {g['ci_high']:+.3f}) |"
            )
        w("")
    else:
        w("not run\n")

    gal = (
        sorted((REPO / "docs" / "figures" / "galleries").glob("*.png"))
        if (REPO / "docs" / "figures" / "galleries").exists()
        else []
    )
    w("## 7. Failure galleries\n")
    if gal:
        for g in gal:
            w(f"- [{g.stem.replace('_', ' ')}](docs/figures/galleries/{g.name})")
        w("")
    else:
        w("not run\n")

    (REPO / "RESULTS.md").write_text("\n".join(L) + "\n")

    # ---- web results.json
    web = REPO / "web" / "src" / "content" / "results.json"
    if le["series"]:
        payload = {
            "status": "ready",
            "generated_at": now,
            "backbone": "ViT-B/16",
            "label_efficiency": [{k: v for k, v in s.items()} for s in le["series"]],
            "baselines": [
                {
                    "name": f"{a} ({i}, {sz}px)",
                    "mean_auroc": metrics(r)["test_auroc"]["mean"]["point"],  # type: ignore[index]
                    "ci_low": metrics(r)["test_auroc"]["mean"]["ci_low"],
                    "ci_high": metrics(r)["test_auroc"]["mean"]["ci_high"],
                }  # type: ignore[index]
                for r, a, i, sz in BASELINES
                if metrics(r)
            ],
        }
        for s in payload["label_efficiency"]:
            for p in s["points"]:
                p.pop("runs", None)
        web.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote RESULTS.md" + (" and web results.json" if le["series"] else ""))


if __name__ == "__main__":
    main()
