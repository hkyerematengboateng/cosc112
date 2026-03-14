"""
evaluator.py
============
Evaluation harness for the Bee Foraging DRL simulation.

Calling conventions are identical to the original evaluator.py so that
existing scripts require no changes:

    run_evaluator(config, scenarios_run)   # runs simulation + plots
    plot_episode_results(scenario_results) # plots from existing results dict
    add_parser_args(model, n, steps)       # builds argparse Namespace
    run_grubi(scenarios)                   # MILP baseline runner

New capabilities added on top of the original:
  - Error bars on all three figures (95 % percentile bootstrap CI on the mean)
  - Significance asterisks (*) on figures where Bonferroni-corrected
    Mann-Whitney U is significant (p_corr < 0.05)
  - StatisticalAnalyzer class:
      · Shapiro-Wilk normality test per (model, timestep) cell
      · Kruskal-Wallis omnibus test across all models
      · Pairwise Mann-Whitney U with Bonferroni correction
      · Cohen's d (parametric) and rank-biserial r (non-parametric) effect sizes
      · 95 % bootstrap CI on the mean
  - print_statistical_report(results)  -- console report
  - export_latex_tables(results, path) -- LaTeX tables for the paper
  - run_evaluator gains optional run_stats / export_latex / latex_path params

CLI (python evaluator.py ...):
    --load results.pkl   skip simulation, load pickle and plot
    --stats              also print full statistical report
    --latex tables.tex   also write LaTeX stat tables
    --scenarios N        number of scenarios per cell  (default 10)
    --models qlearning gradient
    --max_steps 1000     upper bound of timestep sweep
    --steps_interval 100
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

# run_simulation lives in run_test.py -- same import as the original
from run_test import run_simulation
from bees_milp import BeeMILP


# ─────────────────────────────────────────────────────────────────────────────
# Visual identity  (same palette as the original, just centralised)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_COLORS  = ['#2E86AB', '#A23B72', "#0612BE", "#FC0DD8",
                    "#F10808", "#06BE68", '#7B68EE']
_DEFAULT_MARKERS = ['o', 's', '^', 'd', 'v', '<', '*']
_DEFAULT_LINES   = ['-', '-.', ':']
_MILP_COLOR      = "#07B05B"
_POOL_COLOR      = "#040408"

_METRIC_LABELS = {
    "total_pollens_harvested_scen":       "Total Pollen Harvested",
    "total_num_flowers":                  "Total Flowers Harvested",
    "hard_window_flowers_harvested_scen": "Hard-Deadline Flowers Harvested",
    "soft_window_flowers_scen":           "Soft-Deadline Flowers",
    "objective_milp_list":                "MILP Objective Value",
    "total_pollen_per_scenario":          "Total Pollen Available",
}


def _build_model_styles(model_list: List[str]) -> Dict[str, Dict]:
    """Replicate the original per-model style dict."""
    styles = {}
    for i, name in enumerate(model_list):
        display = name.replace('_', '-').title()
        if name == 'qlearning':      display = 'Q-Learning'
        elif name == 'actor_critic': display = 'Actor-Critic'
        elif name == 'gradient':     display = 'Gradient'
        styles[name] = {
            'color':  _DEFAULT_COLORS[(i + 1) % len(_DEFAULT_COLORS)],
            'marker': _DEFAULT_MARKERS[(i + 1) % len(_DEFAULT_MARKERS)],
            'label':  display,
            'lines':  _DEFAULT_LINES[i % len(_DEFAULT_LINES)],
        }
    return styles


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat_obs(step_results: Dict, key: str) -> np.ndarray:
    """
    Return a flat 1-D array of scenario observations for *key* from one
    (model, timestep) cell.  step_results is the dict returned by
    run_simulation() for one step.
    """
    raw = step_results.get(key, [])
    arr = np.asarray(raw, dtype=float).flatten()
    return arr[~np.isnan(arr)]


def _bootstrap_ci(
    obs: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean.  Returns (lower, upper)."""
    if len(obs) < 2:
        mu = float(np.mean(obs)) if len(obs) == 1 else np.nan
        return mu, mu
    rng  = np.random.default_rng(seed)
    boot = np.array([
        np.mean(rng.choice(obs, size=len(obs), replace=True))
        for _ in range(n_bootstrap)
    ])
    a = 1.0 - confidence
    return (float(np.percentile(boot, 100 * a / 2)),
            float(np.percentile(boot, 100 * (1 - a / 2))))


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis engine
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalAnalyzer:
    """
    Full battery of non-parametric statistical tests on DRL evaluation results.

    Input
    -----
    scenario_results : the dict produced by run_evaluator / plot_episode_results
        { model_name: { step_str: { metric_key: [per-scenario values] } } }

    Tests performed per (metric, timestep):
        1. Shapiro-Wilk normality for each model
        2. Kruskal-Wallis omnibus across all models
        3. Pairwise Mann-Whitney U with Bonferroni correction
        4. Cohen's d (parametric effect size)
        5. Rank-biserial r  (non-parametric effect size)
        6. 95 % bootstrap CI on the mean

    Non-parametric tests are used because RL reward distributions are
    routinely non-normal and Shapiro-Wilk detects this formally.
    """

    ALPHA = 0.05

    def __init__(
        self,
        scenario_results: Dict,
        metrics: Optional[List[str]] = None,
        n_bootstrap: int = 2000,
    ):
        self.results     = scenario_results
        self.n_bootstrap = n_bootstrap
        self.metrics     = metrics or [
            "total_pollens_harvested_scen",
            "total_num_flowers",
            "hard_window_flowers_harvested_scen",
        ]
        self.model_list = list(scenario_results.keys())
        first = self.model_list[0]
        self.timesteps = list(scenario_results[first].keys())

    # ── Primitives ────────────────────────────────────────────────────────────

    @staticmethod
    def _shapiro(obs: np.ndarray) -> Tuple[float, float]:
        if len(obs) < 3:
            return np.nan, np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tuple(float(v) for v in scipy_stats.shapiro(obs))

    @staticmethod
    def _kruskal(*groups: np.ndarray) -> Tuple[float, float]:
        valid = [g for g in groups if len(g) >= 2]
        if len(valid) < 2:
            return np.nan, np.nan
        return tuple(float(v) for v in scipy_stats.kruskal(*valid))

    @staticmethod
    def _mwu(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        if len(a) < 2 or len(b) < 2:
            return np.nan, np.nan
        return tuple(
            float(v) for v in
            scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        )

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return np.nan
        pooled = np.sqrt(
            ((len(a) - 1) * np.var(a, ddof=1) +
             (len(b) - 1) * np.var(b, ddof=1))
            / (len(a) + len(b) - 2)
        )
        return 0.0 if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)

    @staticmethod
    def _rbc(u: float, n1: int, n2: int) -> float:
        """Rank-biserial correlation from Mann-Whitney U."""
        if n1 == 0 or n2 == 0:
            return np.nan
        return float(1.0 - (2.0 * u) / (n1 * n2))

    @staticmethod
    def _effect_label(d: float) -> str:
        d = abs(d)
        if np.isnan(d): return "—"
        if d < 0.2:     return "negligible"
        if d < 0.5:     return "small"
        if d < 0.8:     return "medium"
        return "large"

    # ── Full sweep ────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Execute the complete test battery.

        Returns nested dict::

            { metric: { timestep: {
                "normality":   { model: {W, p, normal} },
                "kruskal":     {H, p, significant},
                "pairwise":    { "mA_vs_mB": {U, p_raw, p_corrected,
                                               significant, cohens_d,
                                               r, magnitude} },
                "descriptive": { model: {n, mean, std, median, iqr,
                                          ci95_low, ci95_high} },
            }}}
        """
        pairs         = list(itertools.combinations(self.model_list, 2))
        n_comparisons = max(len(pairs) * len(self.timesteps), 1)
        output: Dict  = {}

        for metric in self.metrics:
            output[metric] = {}
            for ts in self.timesteps:
                obs_map = {
                    m: _flat_obs(self.results[m].get(ts, {}), metric)
                    for m in self.model_list
                }
                cell: Dict = {
                    "normality":   {},
                    "kruskal":     {},
                    "pairwise":    {},
                    "descriptive": {},
                }

                # Descriptive
                for m, obs in obs_map.items():
                    ci_lo, ci_hi = _bootstrap_ci(obs, self.n_bootstrap)
                    q75, q25 = (np.percentile(obs, [75, 25])
                                if len(obs) >= 2 else (np.nan, np.nan))
                    cell["descriptive"][m] = {
                        "n":         len(obs),
                        "mean":      float(np.mean(obs))         if len(obs) > 0 else np.nan,
                        "std":       float(np.std(obs, ddof=1))  if len(obs) > 1 else 0.0,
                        "median":    float(np.median(obs))       if len(obs) > 0 else np.nan,
                        "iqr":       float(q75 - q25),
                        "ci95_low":  ci_lo,
                        "ci95_high": ci_hi,
                    }

                # Shapiro-Wilk
                for m, obs in obs_map.items():
                    W, p = self._shapiro(obs)
                    cell["normality"][m] = {
                        "W": W, "p": p,
                        "normal": (p > self.ALPHA) if not np.isnan(p) else None,
                    }

                # Kruskal-Wallis
                H, p_kw = self._kruskal(*[obs_map[m] for m in self.model_list])
                cell["kruskal"] = {
                    "H": H, "p": p_kw,
                    "significant": bool(p_kw < self.ALPHA) if not np.isnan(p_kw) else False,
                }

                # Pairwise MWU + Bonferroni
                for (ma, mb) in pairs:
                    obs_a, obs_b = obs_map[ma], obs_map[mb]
                    U, p_raw     = self._mwu(obs_a, obs_b)
                    p_corr = min(float(p_raw) * n_comparisons, 1.0) \
                             if not np.isnan(p_raw) else np.nan
                    d = self._cohens_d(obs_a, obs_b)
                    r = self._rbc(U, len(obs_a), len(obs_b))
                    cell["pairwise"][f"{ma}_vs_{mb}"] = {
                        "U":           U,
                        "p_raw":       p_raw,
                        "p_corrected": p_corr,
                        "significant": bool(p_corr < self.ALPHA) if not np.isnan(p_corr) else False,
                        "cohens_d":    d,
                        "r":           r,
                        "magnitude":   self._effect_label(d),
                    }

                output[metric][ts] = cell

        return output

    # ── Console report ────────────────────────────────────────────────────────

    def print_report(self, stats: Dict):
        sep = "─" * 72
        for metric, ts_data in stats.items():
            label = _METRIC_LABELS.get(metric, metric)
            print(f"\n{'═' * 72}\n  METRIC: {label}\n{'═' * 72}")
            for ts, data in ts_data.items():
                print(f"\n  T = {ts}\n  {sep}")
                print("  Descriptive  (mean ± std   [95 % CI]   median   IQR   n):")
                for m, d in data["descriptive"].items():
                    print(
                        f"    {m:20s}  "
                        f"{d['mean']:7.2f} ± {d['std']:6.2f}  "
                        f"[{d['ci95_low']:7.2f}, {d['ci95_high']:7.2f}]  "
                        f"med={d['median']:7.2f}  IQR={d['iqr']:6.2f}  n={d['n']}"
                    )
                print("  Shapiro-Wilk normality:")
                for m, n in data["normality"].items():
                    nstr = ("normal" if n["normal"] else "NON-NORMAL") \
                           if n["normal"] is not None else "—"
                    print(f"    {m:20s}  W={n['W']:.4f}  p={n['p']:.4f}  → {nstr}")
                kw  = data["kruskal"]
                sig = "SIGNIFICANT" if kw["significant"] else "n.s."
                print(f"  Kruskal-Wallis:  H={kw['H']:.4f}  p={kw['p']:.4f}  → {sig}")
                if data["pairwise"]:
                    print("  Pairwise Mann-Whitney U (Bonferroni-corrected):")
                    for pk, pw in data["pairwise"].items():
                        sigstr = "✓ sig." if pw["significant"] else "n.s."
                        print(
                            f"    {pk:35s}  "
                            f"U={pw['U']:.1f}  "
                            f"p_raw={pw['p_raw']:.4f}  "
                            f"p_corr={pw['p_corrected']:.4f}  "
                            f"{sigstr}  "
                            f"d={pw['cohens_d']:.3f} ({pw['magnitude']})  "
                            f"r={pw['r']:.3f}"
                        )

    # ── LaTeX tables ──────────────────────────────────────────────────────────

    def to_latex_descriptive(
        self, stats: Dict, metric: str,
        caption: str = "", label: str = "",
    ) -> str:
        """Descriptive stats table: mean ± std [95 % CI], † = significantly better."""
        ts_data   = stats.get(metric, {})
        timesteps = list(ts_data.keys())
        models    = self.model_list
        col_hdrs  = " & ".join(m.replace("_", "-").title() for m in models)
        met_lbl   = _METRIC_LABELS.get(metric, metric)
        cap = caption or (
            f"Mean $\\pm$ std [95\\,\\% CI] for \\textit{{{met_lbl}}} "
            r"across temporal horizons. "
            r"\dag{} = Bonferroni-corrected Mann-Whitney U "
            r"$p_{\text{corr}}<0.05$ (significantly better than "
            r"$\geq 1$ other model)."
        )
        lbl = label or f"tab:desc_{metric[:20]}"
        lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            f"\\caption{{{cap}}}",
            f"\\label{{{lbl}}}",
            f"\\begin{{tabular}}{{l{'c' * len(models)}}}",
            r"\hline",
            f"\\textbf{{T}} & {col_hdrs} \\\\",
            r"\hline",
        ]
        for ts in timesteps:
            descs  = ts_data[ts]["descriptive"]
            pairs  = ts_data[ts]["pairwise"]
            sig_m  = set()
            for pk, pw in pairs.items():
                if pw["significant"]:
                    ma, mb = pk.split("_vs_")
                    sig_m.add(ma if descs[ma]["mean"] > descs[mb]["mean"] else mb)
            cells = []
            for m in models:
                d   = descs.get(m, {})
                mu  = d.get("mean",     np.nan)
                sd  = d.get("std",      0.0)
                lo  = d.get("ci95_low", np.nan)
                hi  = d.get("ci95_high",np.nan)
                dag = r"\dag{}" if m in sig_m else ""
                cells.append(
                    f"${mu:.1f}\\pm{sd:.1f}${dag}"
                    f"\\newline$[{lo:.1f},{hi:.1f}]$"
                )
            lines.append(f"{ts} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def to_latex_pairwise(
        self, stats: Dict, metric: str, timestep: str,
        caption: str = "", label: str = "",
    ) -> str:
        """Pairwise MWU table for one (metric, timestep)."""
        ts_data   = stats.get(metric, {}).get(timestep, {})
        pairs     = ts_data.get("pairwise", {})
        met_lbl   = _METRIC_LABELS.get(metric, metric)
        cap = caption or (
            f"Pairwise Mann-Whitney U for \\textit{{{met_lbl}}} "
            f"at $T={timestep}$ (Bonferroni, $\\alpha=0.05$). "
            r"Effect sizes: Cohen's $d$, rank-biserial $r$."
        )
        lbl = label or f"tab:mwu_{metric[:15]}_{timestep}"
        lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            f"\\caption{{{cap}}}",
            f"\\label{{{lbl}}}",
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"\textbf{Comparison} & $U$ & $p_{\text{raw}}$ & "
            r"$p_{\text{corr}}$ & Sig. & Cohen's $d$ & $r$ \\",
            r"\hline",
        ]
        fmt = lambda v, s: (s % v) if not np.isnan(v) else "—"
        for pk, pw in pairs.items():
            ma, mb  = pk.split("_vs_")
            la, lb  = ma.replace("_", "-").title(), mb.replace("_", "-").title()
            sym     = r"$\checkmark$" if pw["significant"] else r"$\times$"
            lines.append(
                f"{la} vs.\\ {lb} & "
                f"{fmt(pw['U'],        '%.1f')} & "
                f"{fmt(pw['p_raw'],    '%.4f')} & "
                f"{fmt(pw['p_corrected'],'%.4f')} & "
                f"{sym} & "
                f"{fmt(pw['cohens_d'], '%.3f')} ({pw['magnitude']}) & "
                f"{fmt(pw['r'],        '%.3f')} \\\\"
            )
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def print_statistical_report(scenario_results: Dict) -> Dict:
    """Run full statistical analysis and print the report to stdout."""
    analyzer = StatisticalAnalyzer(scenario_results)
    stats    = analyzer.run()
    analyzer.print_report(stats)
    return stats


def export_latex_tables(
    scenario_results: Dict,
    out_path: str = "stats_tables.tex",
) -> Dict:
    """
    Write LaTeX descriptive + pairwise tables to *out_path*.
    Descriptive tables cover every timestep; pairwise tables use the
    median timestep as a representative example.
    """
    analyzer  = StatisticalAnalyzer(scenario_results)
    stats     = analyzer.run()
    timesteps = list(list(scenario_results.values())[0].keys())
    mid_ts    = timesteps[len(timesteps) // 2]
    blocks    = []
    for metric in analyzer.metrics:
        blocks.append(analyzer.to_latex_descriptive(stats, metric))
        blocks.append("")
        blocks.append(analyzer.to_latex_pairwise(stats, metric, mid_ts))
        blocks.append("")
    Path(out_path).write_text("\n".join(blocks))
    print(f"[evaluator] LaTeX tables written → {out_path}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Significance annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_significance(
    ax,
    x_labels: list,
    model_list: List[str],
    stats: Optional[Dict],
    metric: str,
):
    """
    Place a bold '*' above x positions where at least one Bonferroni-
    corrected pairwise test is significant for the given metric.
    """
    if stats is None or metric not in stats:
        return
    ylim  = ax.get_ylim()
    y_top = ylim[1] - 0.02 * (ylim[1] - ylim[0])
    for xi, ts in enumerate(x_labels):
        pairs = stats[metric].get(str(ts), {}).get("pairwise", {})
        if any(pw["significant"] for pw in pairs.values()):
            ax.text(xi, y_top, "*", ha="center", va="top",
                    fontsize=13, fontweight="bold", color="#222222",
                    transform=ax.transData)


# ─────────────────────────────────────────────────────────────────────────────
# Error-bar line helper
# ─────────────────────────────────────────────────────────────────────────────

def _errorbar_series(
    ax,
    x: list,
    means: list,
    scenario_results: Dict,
    model_name: str,
    step_interval: list,
    metric_key: str,
    style: Dict,
    n_bootstrap: int = 2000,
):
    """
    Plot a line with 95 % bootstrap CI error bars (capped, publication style).
    Falls back to a plain line if fewer than 2 observations exist per cell.
    """
    ci_lo, ci_hi = [], []
    for step in step_interval:
        obs      = _flat_obs(scenario_results[model_name].get(step, {}), metric_key)
        lo, hi   = _bootstrap_ci(obs, n_bootstrap)
        ci_lo.append(lo)
        ci_hi.append(hi)

    means_arr = np.asarray(means, dtype=float)
    yerr_lo   = np.clip(means_arr - np.array(ci_lo), 0, None)
    yerr_hi   = np.clip(np.array(ci_hi) - means_arr, 0, None)

    ax.errorbar(
        x, means,
        yerr=[yerr_lo, yerr_hi],
        fmt=style['marker'],
        linestyle=style['lines'],
        linewidth=2,
        markersize=8,
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        color=style['color'],
        alpha=0.8,
        label=f"{style['label']} Model",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_episode_results(
    scenario_results: Dict[str, Any],
    run_stats: bool = True,
    n_bootstrap: int = 2000,
) -> Optional[Dict]:
    """
    Generate all three comparison figures with 95 % bootstrap CI error bars
    and significance annotations.

    Parameters
    ----------
    scenario_results : dict
        { model_name: { step_str: { metric_key: [per-scenario values] } } }
        Identical format to the original evaluator.
    run_stats : bool
        If True, run the full statistical analysis and attach significance
        markers (*) to the figures.
    n_bootstrap : int
        Number of bootstrap resamples for CI estimation.

    Returns
    -------
    stats : dict or None
    """
    # Remove MILP dict if present (original behaviour)
    scenario_results.pop('grubi_result_list', None)

    model_list    = list(scenario_results.keys())
    step_interval = list(scenario_results[model_list[0]].keys())
    model_styles  = _build_model_styles(model_list)

    # Run statistical analysis
    stats = None
    if run_stats:
        analyzer = StatisticalAnalyzer(scenario_results, n_bootstrap=n_bootstrap)
        stats    = analyzer.run()

    # Build mean series per model per step (mirrors original loop structure)
    model_means: Dict[str, Dict[str, Dict]] = {}
    for model in model_list:
        model_means[model] = {}
        for step in step_interval:
            sr = scenario_results[model].get(step, {})

            def _mean(k):
                obs = _flat_obs(sr, k)
                return float(np.mean(obs)) if len(obs) > 0 else 0.0

            model_means[model][step] = {
                'total_pollens':     _mean('total_pollens_harvested_scen'),
                'total_pollen_scen': _mean('total_pollen_per_scenario'),
                'milp_obj':          _mean('objective_milp_list'),
                'total_flowers':     _mean('total_num_flowers'),
                'hard_harvested':    _mean('hard_window_flowers_harvested_scen'),
                'hard_total':        _mean('hard_window_flowers_scen'),
            }

    first = model_list[0]
    v = [model_means[first][s]['milp_obj']          for s in step_interval]
    p = [model_means[first][s]['total_pollen_scen'] for s in step_interval]
    x = step_interval

    # ── Figure 1: Total pollen harvested ─────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(x, p, marker='*', linestyle='-',  linewidth=2, markersize=8,
             color=_POOL_COLOR, alpha=0.8, label='Average Total Pollen')
    ax1.plot(x, v, marker='*', linestyle='-.', linewidth=2, markersize=8,
             color=_MILP_COLOR, alpha=0.8, label='MILP Solver')
    for model in model_list:
        means = [model_means[model][s]['total_pollens'] for s in step_interval]
        _errorbar_series(ax1, x, means, scenario_results, model,
                         step_interval, 'total_pollens_harvested_scen',
                         model_styles[model], n_bootstrap)
    _annotate_significance(ax1, step_interval, model_list, stats,
                           'total_pollens_harvested_scen')
    ax1.set_xlabel('Max. Number of Time Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count of Flower Pollens Harvested', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Total Pollens Harvested Compared with MILP Solver\n'
        '(error bars = 95 % bootstrap CI;  * = Bonferroni-corrected MWU p < 0.05)',
        fontsize=13, fontweight='bold', pad=16,
    )
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('images/total_pollen_harvested_comparison_new_best.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Total flowers harvested ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(x, [100] * len(step_interval),
             marker='*', linestyle='-.', linewidth=2, markersize=8,
             color=_MILP_COLOR, alpha=0.8, label='Total Flowers in Scenario')
    for model in model_list:
        means = [model_means[model][s]['total_flowers'] for s in step_interval]
        _errorbar_series(ax2, x, means, scenario_results, model,
                         step_interval, 'total_num_flowers',
                         model_styles[model], n_bootstrap)
    _annotate_significance(ax2, step_interval, model_list, stats, 'total_num_flowers')
    ax2.set_xlabel('Max. Number of Time Steps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count of Flowers Harvested', fontsize=12, fontweight='bold')
    ax2.set_title(
        'Comparison of Total Flowers Harvested\n'
        '(error bars = 95 % bootstrap CI;  * = Bonferroni-corrected MWU p < 0.05)',
        fontsize=13, fontweight='bold', pad=16,
    )
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    # plt.savefig('images/flowers_harvested_comparison.png', ...)  # kept commented as original
    plt.close()

    # ── Figure 3: Hard-deadline flowers harvested ─────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    hard_ref = [model_means[first][s]['hard_total'] for s in step_interval]
    ax3.plot(x, hard_ref, marker='*', linestyle='-.', linewidth=2, markersize=8,
             color=_MILP_COLOR, alpha=0.8, label='Total Hard Time-Window Flowers')
    for model in model_list:
        means = [model_means[model][s]['hard_harvested'] for s in step_interval]
        _errorbar_series(ax3, x, means, scenario_results, model,
                         step_interval, 'hard_window_flowers_harvested_scen',
                         model_styles[model], n_bootstrap)
    _annotate_significance(ax3, step_interval, model_list, stats,
                           'hard_window_flowers_harvested_scen')
    ax3.set_xlabel('Max. Number of Time Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count of Hard Time-Step Flowers', fontsize=12, fontweight='bold')
    ax3.set_title(
        'Hard-Deadline Flowers Harvested by Model\n'
        '(error bars = 95 % bootstrap CI;  * = Bonferroni-corrected MWU p < 0.05)',
        fontsize=13, fontweight='bold', pad=16,
    )
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig('images/hard_flowers_harvested_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# plot_results  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results):
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(1, len(results['gradient']) + 1)
    ax.plot(x, results['gradient'], 'o-', linewidth=2, markersize=8,
            label='Gradient Method', color='#2E86AB', alpha=0.8)
    ax.plot(x, results['qlearning'], 's-', linewidth=2, markersize=8,
            label='Q-Learning', color='#A23B72', alpha=0.8)
    ax.set_xlabel('Run Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Result Value', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Gradient Method vs Q-Learning Results',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_xticks(x)
    stats_text = (
        "Statistics:\n"
        f"  Gradient:   Mean={np.mean(results['gradient']):.2f}  "
        f"Std={np.std(results['gradient']):.2f}\n"
        f"  Q-Learning: Mean={np.mean(results['qlearning']):.2f}  "
        f"Std={np.std(results['qlearning']):.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# add_parser_args  (identical signature to the original)
# ─────────────────────────────────────────────────────────────────────────────

def add_parser_args(model: str, num_of_scenario: int, step_count: int):
    """
    Build an argparse Namespace exactly as the original did.
    Uses parse_args([]) so it never reads sys.argv when called programmatically.
    """
    parser = argparse.ArgumentParser(description='Bee Forage Simulation Evaluator')
    parser.add_argument("-p",  "--policy",           default=model,
                        choices=['gradient', 'qlearning'])
    parser.add_argument("-e",  "--num_of_scenarios", default=num_of_scenario)
    parser.add_argument("-r",  "--render",           default=None)
    parser.add_argument("-m",  "--max_steps",        default=step_count)
    parser.add_argument("-b",  "--bees",             default=None)
    parser.add_argument("-f",  "--flowers",          default=None)
    parser.add_argument("-c",  "--max_bee_capacity", default=100)
    parser.add_argument("-el", "--max_energy_level", default=2000)
    parser.add_argument("-g",  "--grid_size",        default=None)
    return parser.parse_args([])


# ─────────────────────────────────────────────────────────────────────────────
# run_evaluator  (extends original; backward-compatible signature)
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluator(
    config: Dict[str, Any],
    scenarios_run: int,
    run_stats: bool = True,
    export_latex: bool = False,
    latex_path: str = "stats_tables.tex",
    results_pkl: str = "results_m_steps_v2.pkl",
):
    """
    Run the full evaluation sweep, persist results, plot figures, and
    optionally print statistical analysis and export LaTeX tables.

    Parameters
    ----------
    config : dict
        Must contain 'models' key: list of model name strings.
        Optional keys: 'max_steps' (int, default 1000),
                       'steps_interval' (int, default 100).
    scenarios_run : int
        Number of scenarios per model × timestep cell.
    run_stats : bool
        Run StatisticalAnalyzer after plotting and print the report.
    export_latex : bool
        Write LaTeX stat tables to *latex_path*.
    results_pkl : str
        Path for the pickled results file.
    latex_path : str
        Output path for LaTeX tables (used when export_latex=True).
    """
    models         = config['models']
    max_steps      = int(config.get('max_steps', 1000))
    steps_interval = int(config.get('steps_interval', 100))

    final_results: Dict = {}
    for model in models:
        steps_results: Dict = {}
        for step_count in range(steps_interval, max_steps + steps_interval, steps_interval):
            args    = add_parser_args(model, scenarios_run, step_count)
            rewards = run_simulation(args)
            steps_results[str(step_count)] = rewards
        final_results[model] = steps_results

    # Save
    pkl_path = Path(results_pkl)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'wb') as fh:
        pickle.dump(final_results, fh)
    print(f"[evaluator] Results saved → {pkl_path}")

    # Plot with error bars
    stats = plot_episode_results(final_results, run_stats=run_stats)

    if run_stats and stats is not None:
        analyzer = StatisticalAnalyzer(final_results)
        analyzer.print_report(stats)

    if export_latex and stats is not None:
        export_latex_tables(final_results, latex_path)

    return final_results, stats


# ─────────────────────────────────────────────────────────────────────────────
# run_grubi  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def run_grubi(scenarios: int):
    milp = BeeMILP()
    objective_list = []
    test_file_name = (
        "datasets/test_data_episodes_10_gridsize_20_flowers_300"
        "_bees_15_steps_800.toml"
    )
    for i in range(scenarios):
        result = milp.analyze_scenario_dataset(test_file_name, i + 1)
        objective_list.append(result)
    return objective_list


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point  (extends original __main__ block)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description="Bee Foraging DRL Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("--scenarios",      type=int, default=10)
    cli.add_argument("--models",         nargs="+", default=["qlearning", "gradient"])
    cli.add_argument("--max_steps",      type=int, default=1000)
    cli.add_argument("--steps_interval", type=int, default=100)
    cli.add_argument("--results_pkl",    default="results_m_steps_v2.pkl")
    cli.add_argument("--load",           action="store_true",
                     help="Skip simulation; load --results_pkl and plot.")
    cli.add_argument("--stats",          action="store_true",
                     help="Print full statistical report to stdout.")
    cli.add_argument("--latex",          default=None,
                     help="Export LaTeX stat tables to this path.")
    args = cli.parse_args()

    if args.load:
        print(f"[evaluator] Loading {args.results_pkl} …")
        with open(args.results_pkl, "rb") as fh:
            final_results = pickle.load(fh)
        stats = plot_episode_results(final_results, run_stats=True)
    else:
        config = {
            'models':         args.models,
            'max_steps':      args.max_steps,
            'steps_interval': args.steps_interval,
        }
        final_results, stats = run_evaluator(
            config,
            scenarios_run = args.scenarios,
            run_stats     = True,
            export_latex  = args.latex is not None,
            latex_path    = args.latex or "stats_tables.tex",
            results_pkl   = args.results_pkl,
        )

    if args.stats and stats is not None:
        analyzer = StatisticalAnalyzer(final_results)
        analyzer.print_report(stats)

    if args.latex and stats is not None:
        export_latex_tables(final_results, args.latex)