"""Regenerate every table and figure from results/ (`uv run python run.py report`).

Local: reads the JSON / .pt outputs written by the phases, writes markdown tables to
results/tables/*.md and PNG/PDF figures to figures/, and injects tables into README.md and
the paper source between `<!-- TABLE:name -->` ... `<!-- /TABLE:name -->` markers, so no number
in the write-ups is typed by hand.

Global context: chart styling follows the dataviz reference palette (categorical slots in
fixed order, one-hue sequential ramp, a cyclic map only for circular quantities, one y-axis
per panel, recessive grid).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from modsae.config import ROOT, Paths  # noqa: E402

# dataviz reference palette (light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("ref_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
P = 113


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": TEXT, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _save(fig, name: str) -> None:
    out = Paths().make().figures
    fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.2e}" if (abs(x) < 10 ** (-nd) and x != 0) else f"{x:.{nd}f}"
    return str(x)


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write_table(name: str, text: str) -> None:
    out = Paths().make().results / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(text, encoding="utf-8")


def inject(path: Path, tables: dict[str, str]) -> None:
    """Replace the content between <!-- TABLE:name --> and <!-- /TABLE:name --> markers."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for name, body in tables.items():
        pat = re.compile(rf"(<!-- TABLE:{name} -->\n).*?(\n?<!-- /TABLE:{name} -->)", re.S)
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- tables

def mechanism_table(results: Path) -> str:
    rows = []
    for run_dir in sorted(results.iterdir()):
        m = _load(run_dir / "mechanism.json")
        if not m:
            continue
        rr = m.get("resid_mid_regression", {})
        r2_ab = sum(v["r2_a_plus_b"] for v in rr.values()) if rr else None
        sv = max((v["sv3_over_sv1"] for v in rr.values()), default=None)
        rows.append([m["run"], ", ".join(map(str, m["K"])), m["n_single_frequency_neurons"], m["clean"]["test"][1],
                     m.get("restricted", {}).get("test", [None, None])[1], m.get("excluded", {}).get("test", [None, None])[1],
                     r2_ab, sv, m["ab_asymmetry"]["resid_mid"], m["gradient_symmetricity"], m["distance_irrelevance"], m["included"]])
    header = ["model", "key freqs K", "single-freq neurons (/512)", "test acc", "restricted test acc", "excluded test acc",
              "resid_mid R² on A+B features (sum over K)", "max σ3/σ1 (rank-2 check)", "a↔b asymmetry", "grad. symmetricity", "distance irrelevance", "included"]
    return md_table(header, rows)


def sae_table(results: Path, base_run: str | None = None) -> str:
    rows = []
    for run_dir in sorted(results.iterdir()):
        m = _load(run_dir / "metrics.json")
        if not m or (base_run and m["config"]["base_run"] != base_run):
            continue
        c = _load(run_dir / "characterization.json") or {}
        cfg = m["config"]
        rows.append([cfg["site"], cfg["arch"], cfg["d_sae"], cfg["k"], cfg["seed"], m["test_fvu"], m["test_logit_fvu"], m["test_l0"],
                     m["dead_fraction"], m["spliced"]["acc"], m["spliced"]["kl"], c.get("n_alive"), c.get("H1_fraction_ring"),
                     c.get("median_r2_key_best"), c.get("median_width_ring")])
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))
    header = ["site", "arch", "m", "k", "SAE seed", "test FVU", "test logit FVU", "test L0", "dead frac", "spliced acc", "spliced KL",
              "alive", "ring-feature frac", "median best-key arc R²", "median ring width"]
    return md_table(header, rows)


# ---------------------------------------------------------------- figures

def fig_training(results: Path, run: str = "base_s999") -> None:
    h = _load(results / run / "training.json")
    if not h:
        return
    ep = np.array(h["epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, label, color in (("train_loss", "train", SERIES[0]), ("test_loss", "test", SERIES[1])):
        axes[0].plot(ep, h[key], color=color, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss (log scale)")
    axes[0].set_title(f"{run}: loss")
    for key, label, color in (("train_acc", "train", SERIES[0]), ("test_acc", "test", SERIES[1])):
        axes[1].plot(ep, h[key], color=color, label=label)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{run}: accuracy")
    for ax in axes:
        ax.legend(loc="center right")
    _save(fig, f"training_{run}")


def fig_resid_geometry(run: str = "base_s999") -> None:
    """Fitted 2-D plane of resid_mid for each key frequency, coloured by the sum (cyclic map)."""
    from modsae import data, mechanism

    grid_path = Paths().cache / run / "grid.pt"
    mech = _load(Paths().results / run / "mechanism.json")
    if not grid_path.exists() or not mech or not mech["K"]:
        return
    X = torch.load(grid_path, weights_only=True)["resid_mid"]
    a, b = data.index_to_ab(torch.arange(P * P), P)
    s = ((a + b) % P).numpy()
    K = mech["K"]
    fig, axes = plt.subplots(1, len(K), figsize=(3.4 * len(K), 3.4), constrained_layout=True)
    for i, (ax, k) in enumerate(zip(np.atleast_1d(axes), K)):
        z = (X - X.mean(0)) @ mechanism.fitted_plane(X, k, P).T
        # colour by the frequency-k phase of the sum so one revolution of colour = one revolution of the circle
        phase = (k * s) % P
        sc = ax.scatter(z[:, 0], z[:, 1], c=phase, cmap="twilight", s=2, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("plane axis 1")
        if i == 0:
            ax.set_ylabel("plane axis 2")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8)
    cb.set_label("k·(a+b) mod 113 (cyclic)")
    fig.suptitle(f"{run}: resid_mid in each key-frequency plane; colour = k(a+b) mod 113 (two colour cycles per turn = half-angle)", color=TEXT)
    _save(fig, f"resid_geometry_{run}")


def fig_sweep(results: Path, base_run: str = "base_s999") -> None:
    """Heatmaps over (k, m) for each site: val FVU, dead fraction, ring-feature fraction, median best-key R²."""
    rows = []
    for run_dir in results.iterdir():
        m = _load(run_dir / "metrics.json")
        c = _load(run_dir / "characterization.json")
        if not m or not c or m["config"]["base_run"] != base_run or m["config"]["arch"] != "batchtopk" or m["config"]["seed"] != 0 or m["config"]["variant"]:
            continue
        rows.append((m["config"]["site"], m["config"]["k"], m["config"]["d_sae"], m.get("val", {}).get("fvu", m["test_fvu"]),
                     m["dead_fraction"], c["H1_fraction_ring"], c.get("median_r2_key_best") or np.nan, m.get("val", {}).get("spliced", {}).get("acc", np.nan)))
    if not rows:
        return
    sites = sorted({r[0] for r in rows})
    ks = sorted({r[1] for r in rows})
    ms = sorted({r[2] for r in rows})
    metrics_ = [("val FVU", 3, BLUES), ("dead fraction", 4, BLUES), ("ring-feature fraction", 5, BLUES), ("median best-key arc R²", 6, BLUES), ("val spliced acc", 7, BLUES)]
    fig, axes = plt.subplots(len(sites), len(metrics_), figsize=(3.3 * len(metrics_), 2.9 * len(sites)), squeeze=False)
    for i, site in enumerate(sites):
        for j, (title, col, cmap) in enumerate(metrics_):
            M = np.full((len(ks), len(ms)), np.nan)
            for r in rows:
                if r[0] == site:
                    M[ks.index(r[1]), ms.index(r[2])] = r[col]
            ax = axes[i, j]
            im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=max(1e-9, np.nanmax(M)) if col == 3 else 1)
            ax.set_xticks(range(len(ms)), ms)
            ax.set_yticks(range(len(ks)), ks)
            ax.set_xlabel("dictionary size m")
            ax.set_ylabel("k (target L0)")
            ax.set_title(f"{site}: {title}")
            ax.grid(False)
            for (yy, xx), v in np.ndenumerate(M):
                if not np.isnan(v):
                    ax.text(xx, yy, f"{v:.2f}" if v >= 0.01 else f"{v:.0e}", ha="center", va="center", fontsize=7,
                            color="white" if (v / (np.nanmax(M) if col == 3 else 1)) > 0.55 else TEXT)
            fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, f"sweep_{base_run}")


def fig_feature_gallery(run_id: str, n: int = 8) -> None:
    """Activation maps over the (a, b) grid for the alive features with the highest arc R², plus their ψ tuning."""
    from modsae import interventions, mechanism, phases
    from modsae.config import device

    fpath = Paths().results / run_id / "features.pt"
    cfg = _load(Paths().results / run_id / "metrics.json")
    if not fpath.exists() or not cfg:
        return
    feat = torch.load(fpath, weights_only=False)
    table, a_idx = feat["table"], feat["alive_idx"]
    if "r2_key_best" not in table:
        return
    site, base_run = cfg["config"]["site"], cfg["config"]["base_run"]
    grid = phases.grid_cache(base_run, device())
    sae = phases.load_sae_for(run_id, device())
    Z = interventions.encode(sae, grid[site])
    order = torch.argsort(torch.nan_to_num(table["r2_key_best"], nan=-1), descending=True)[:n]
    fig, axes = plt.subplots(2, len(order), figsize=(2.3 * len(order), 4.8), squeeze=False)
    for col, j in enumerate(order.tolist()):
        f = Z[:, a_idx[j]].reshape(P, P).numpy()
        ax = axes[0, col]
        ax.imshow(f, cmap=BLUES, origin="lower")
        ax.set_title(f"#{int(a_idx[j])} k={int(table['k_key'][j])}\nR²={table['r2_key_best'][j]:.2f}", fontsize=8)
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.grid(False)
        psi, r = mechanism.analytic_coords(P, int(table["k_key"][j]))
        ax2 = axes[1, col]
        ax2.scatter(psi, Z[:, a_idx[j]] / r.clamp_min(1e-3) ** float(table["gamma"][j]), s=1, color=SERIES[0], rasterized=True)
        ax2.set_xlabel("ψ_k")
        ax2.set_ylabel("activation / r^γ")
    fig.suptitle(f"{run_id}: top features by arc-model R²", color=TEXT, fontsize=10)
    _save(fig, f"features_{run_id}")


def main() -> None:
    style()
    paths = Paths().make()
    results = paths.results
    tables = {"mechanism": mechanism_table(results), "sae_dev": sae_table(results, "base_s999")}
    for name, text in tables.items():
        write_table(name, text)
    fig_training(results)
    fig_resid_geometry()
    fig_sweep(results)
    inject(ROOT / "README.md", tables)
    inject(ROOT / "reports" / "paper" / "paper.typ", tables)
    print("report written:", sorted(p.name for p in (results / "tables").iterdir()))


if __name__ == "__main__":
    main()
