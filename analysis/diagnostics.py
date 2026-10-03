"""
Model diagnostics for the pooled ladder — assumptions + error metrics,
numeric (console) and graphical (self-contained HTML with inline SVG).

EVERY NUMBER COMES FROM THE FITTED MODELS estimate.py REPORTS — nothing is
re-implemented in parallel. Rung 1 is re-fit through the identical
adjusted_coef/_design machinery (and asserted equal); rungs 2-4 are the very
estimator objects from estimate.fit_pooled_dml (the same constructor
run_pooled_dml prints from), fit with cache_values=True so econml exposes its
own cross-fitted first-stage residuals (.residuals_).

WHAT IS CHECKED HERE (and where the untestable assumptions are delegated):

  * rung 1 pooled OLS — fit quality (R^2, adjusted R^2, residual sd),
    residual-vs-fitted and QQ plots.
  * DML FIRST STAGE — out-of-fold R^2 of E[Y|W] and E[T|W] implied by each
    fitted rung's own residuals (incl. what 'auto' actually selected), with
    the rung's reported ATE alongside.
  * FUNCTIONAL FORM — the linear final stage assumes a single slope in HDI.
    Checked on the pre-registered rung-3 estimator's own residualized scale:
    a quadratic-term test (the U-shape audit, numeric) and decile-binned
    means on the Robinson plot (graphical), whose slope IS the rung-3 ATE.
  * POSITIVITY / IDENTIFYING VARIATION — how much treatment variance survives
    the adjustment set, overall and per region (the DML self-localization
    weights, from rung 3's residualized treatment), numeric + bar chart.
  * NOT HERE — exchangeability is untestable and is probed by refute.py
    (battery, negative control, E-values); honest clustered inference is
    reported by estimate.py (origin-clustered SEs, dof = G-1, LOO).

No pass/fail thresholds are introduced here — these are descriptive
diagnostics; the pre-registered criteria live in refute.py.

Usage:   uv run python analysis/diagnostics.py --run
Output:  analysis/outputs/model_diagnostics.html (+ console summary)
"""
from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from analysis.estimate import (CONTRAST, DML_CV, SEED, TUNING_JSON,
                                   _design, adjusted_coef, fit_pooled_dml,
                                   load_frame)
except ImportError:
    from estimate import (CONTRAST, DML_CV, SEED, TUNING_JSON,
                          _design, adjusted_coef, fit_pooled_dml,
                          load_frame)

OUT = Path(__file__).resolve().parent / "outputs" / "model_diagnostics.html"

# dark palette to match dag.html / positivity_atlas.html
BG, CARD, FG, MUT = "#0b0f1a", "#121829", "#e7ecf5", "#9aa6bd"


def _style(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color("#26314e")
    ax.tick_params(colors=MUT, labelsize=9)
    ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)
    ax.title.set_color(FG)
    ax.grid(color="#1e2740", linewidth=0.6)


def _legend(ax, **kw):
    """Legend styled for the dark theme (matplotlib's default is a white box
    that swallows the light label text)."""
    return ax.legend(facecolor="#1a2238", edgecolor="#26314e",
                     labelcolor=FG, framealpha=0.95, **kw)


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _binned(x, y, bins=10):
    """Decile-binned means of y over x: (centers, means, 2*SE)."""
    q = np.quantile(x, np.linspace(0, 1, bins + 1))
    q[0] -= 1e-12
    idx = np.digitize(x, q[1:-1])
    cx, my, se2 = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() < 3:
            continue
        cx.append(float(x[m].mean())); my.append(float(y[m].mean()))
        se2.append(2 * float(y[m].std(ddof=1) / np.sqrt(m.sum())))
    return np.array(cx), np.array(my), np.array(se2)


def _hc1_quadratic_p(t_res, y_res):
    """p-value (HC1) of the quadratic term in y_res ~ t_res + t_res^2 — the
    functional-form / U-shape check on the partialled-out scale."""
    X = np.column_stack([np.ones_like(t_res), t_res, t_res ** 2])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y_res)
    r = y_res - X @ b
    n, k = X.shape
    V = (n / (n - k)) * (XtX_inv @ ((X * (r ** 2)[:, None]).T @ X) @ XtX_inv)
    tstat = b[2] / np.sqrt(max(V[2, 2], 1e-300))
    return float(b[2]), float(2 * stats.t.sf(abs(tstat), n - k))


RUNG = {"linear": "rung 2 — linear",
        "auto": "rung 3 — 'auto' (pre-registered)",
        "tuned": "rung 4 — Optuna-tuned"}


def _inner_model_names(models) -> str:
    """Best-effort: the model classes econml actually fitted per fold — shows
    what the 'auto' bake-off selected. Wrapper attribute names vary, so this
    walks common ones and falls back to the wrapper's own class name."""
    def leaf(m, depth=0):
        if depth > 6:
            return type(m).__name__
        for attr in ("best_model", "_model", "model"):
            inner = getattr(m, attr, None)
            if inner is not None and inner is not m and not isinstance(inner, str):
                return leaf(inner, depth + 1)
        return type(m).__name__
    try:
        flat, stack = [], list(models)
        while stack:
            m = stack.pop(0)
            if isinstance(m, (list, tuple)):
                stack.extend(m)
            else:
                flat.append(leaf(m))
        uniq = sorted(set(flat))
        return ", ".join(f"{u} ({flat.count(u)}/{len(flat)} folds)" for u in uniq)
    except Exception:
        return "n/a"


def run(outcome: str = "y_first_trip") -> int:
    df = load_frame()

    # ---- rung 1: the exact estimate.py estimator (adjusted_coef) ---------- #
    d1 = df.dropna(subset=[outcome]).copy()
    d1["yr"] = d1.trip_year.astype(int).astype(str)
    dd = d1.rename(columns={outcome: "_y"})
    beta1, se1, ci1, p1, dof1 = adjusted_coef(
        dd, outcome="_y", treatment="hdi", numeric=[],
        categorical=["region", "yr"], cluster_col="origin_iso", ci_level=0.95)
    X1, names1 = _design(dd, "hdi", [], ["region", "yr"])
    yv = dd["_y"].to_numpy(float)
    bvec = np.linalg.pinv(X1.T @ X1) @ (X1.T @ yv)
    # same design, same solver as adjusted_coef — must agree exactly
    assert abs(float(bvec[names1.index("hdi")]) - beta1) < 1e-8
    fitted = X1 @ bvec
    resid = yv - fitted
    n = len(dd)
    r2 = 1 - float((resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - X1.shape[1])
    resid_sd = float(resid.std(ddof=X1.shape[1]))

    # ---- rungs 2-4: the ACTUAL fitted estimators (estimate.fit_pooled_dml) - #
    specs = ["linear", "auto"] + (["tuned"] if TUNING_JSON.exists() else [])
    rungs = []
    for spec in specs:
        est, d = fit_pooled_dml(df, outcome, nuisances=spec, cache_values=True)
        ce = est.estimator.estimator            # the fitted econml LinearDML
        y_res, t_res, _, W_fit = ce.residuals_  # ITS cross-fitted residuals
        y_res = np.asarray(y_res, float).ravel()
        t_res = np.asarray(t_res, float).ravel()
        yv2, tv2 = d[outcome].to_numpy(float), d.hdi.to_numpy(float)
        rungs.append({
            "spec": spec, "name": RUNG[spec], "ate": float(est.value),
            "r2_y": 1 - float((y_res ** 2).sum() / ((yv2 - yv2.mean()) ** 2).sum()),
            "r2_t": 1 - float((t_res ** 2).sum() / ((tv2 - tv2.mean()) ** 2).sum()),
            "y_res": y_res, "t_res": t_res, "d": d,
            "W_fit": np.asarray(W_fit, float),
            "selected": _inner_model_names(ce.models_y) if spec == "auto" else ""})

    # the pre-registered rung (3, 'auto') carries the graphical diagnostics
    r3 = next(r for r in rungs if r["spec"] == "auto")
    d3, yt, tt = r3["d"], r3["y_res"], r3["t_res"]
    region = d3.region.to_numpy()
    # row-alignment check: econml's cached W must be constant within each
    # (region, year) cell of OUR frame — true only if its rows are in frame
    # order. (A direct array comparison is too brittle: DoWhy's dummy
    # encoding differs from ours in column order/reference levels.)
    codes = (d3.region + "|" + d3.trip_year.astype(int).astype(str)).to_numpy()
    order_ok = all(np.allclose(r3["W_fit"][codes == c], r3["W_fit"][codes == c][0])
                   for c in set(codes))
    slope = float((tt * yt).sum() / (tt * tt).sum())  # == rung-3 ATE / CONTRAST
    tv3 = d3.hdi.to_numpy(float)
    sd_t, sd_tt = float(tv3.std()), float(tt.std())
    quad_coef, quad_p = _hc1_quadratic_p(tt, yt)

    regs = sorted(set(region), key=lambda r: -(region == r).sum())
    rows = []
    for r in regs:
        m = region == r
        rows.append((r, int(m.sum()), float(tt[m].std()),
                     m.sum() * float(tt[m].var())))
    tot_w = sum(w for *_, w in rows) or 1.0
    rows = [(r, nn, s, w / tot_w) for r, nn, s, w in rows]

    # ---- console summary --------------------------------------------------- #
    slope_match = abs(slope * CONTRAST - r3["ate"]) < 5e-4
    print(f"\nMODEL DIAGNOSTICS — pooled tier, outcome={outcome} (n={n:,})")
    print("  (every number below comes from the fitted objects estimate.py reports)")
    print(f"  rung 1 OLS (adjusted_coef): beta per +{CONTRAST:.2f} = "
          f"{beta1 * CONTRAST:+.4f} (p={p1:.3f}, dof={dof1})  "
          f"R2={r2:.4f}  adj R2={adj_r2:.4f}  resid sd={resid_sd:.4f}")
    print(f"  rungs 2-4 (fitted LinearDML internals, {DML_CV}-fold cross-fit, seed {SEED}):")
    print(f"    {'rung':<32} {'ATE/+' + format(CONTRAST, '.2f'):>10} "
          f"{'R2 E[Y|W]':>10} {'R2 E[T|W]':>10}")
    for r in rungs:
        print(f"    {r['name']:<32} {r['ate']:>+10.4f} "
              f"{r['r2_y']:>10.4f} {r['r2_t']:>10.4f}")
        if r["selected"]:
            print(f"    {'':<32} 'auto' selected: {r['selected']}")
    print(f"  Robinson slope from rung-3's OWN residuals: {slope:+.4f}/unit = "
          f"{slope * CONTRAST:+.4f} per +{CONTRAST:.2f} "
          f"({'matches' if slope_match else 'MISMATCH vs'} the rung-3 ATE {r3['ate']:+.4f})")
    print(f"  functional form (rung-3 residual scale): quadratic coef={quad_coef:+.3f}, "
          f"p={quad_p:.3f} (HC1) — the U-shape audit")
    print(f"  identifying variation: sd(HDI)={sd_t:.4f} -> sd(HDI|W)={sd_tt:.4f} "
          f"({sd_tt**2 / sd_t**2:.1%} of variance survives the adjustment set)")
    if not order_ok:
        print("  [warn] econml's cached W does not match the frame order — "
              "region overlays skipped")
    print("  DML weight share (n_r x Var(T_res|W)) by region (rung 3):")
    for r, nn, s_, w in rows:
        print(f"    {r:<20} n={nn:>4}  sd(T_res)={s_:.4f}  weight={w:.1%}")

    # ---- figures ------------------------------------------------------------ #
    plt.rcParams.update({"text.color": FG, "font.size": 10})
    figs = {}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=BG)
    ax = axes[0]; _style(ax)
    ax.scatter(fitted, resid, s=6, alpha=0.25, color="#3b82f6", linewidths=0)
    bx, bm, bs = _binned(fitted, resid)
    ax.errorbar(bx, bm, yerr=bs, fmt="o-", color="#f59e0b", ms=4, lw=1.2,
                label="decile means ± 2SE")
    ax.axhline(0, color="#ef4444", lw=1)
    ax.set_xlabel("fitted"); ax.set_ylabel("residual")
    ax.set_title("rung 1 OLS — residual vs fitted"); _legend(ax, fontsize=9)
    ax = axes[1]; _style(ax)
    (osm, osr), (sl, ic, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=6, alpha=0.3, color="#3b82f6", linewidths=0)
    ax.plot(osm, sl * osm + ic, color="#ef4444", lw=1)
    ax.set_xlabel("normal quantiles"); ax.set_ylabel("residual quantiles")
    ax.set_title("rung 1 OLS — residual QQ")
    figs["ols"] = _svg(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5), facecolor=BG); _style(ax)
    if order_ok:
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(regs):
            m = region == r
            ax.scatter(tt[m], yt[m], s=7, alpha=0.35, linewidths=0,
                       color=cmap(i % 10), label=f"{r} (n={m.sum()})")
    else:
        ax.scatter(tt, yt, s=7, alpha=0.3, linewidths=0, color="#3b82f6")
    bx, bm, bs = _binned(tt, yt)
    ax.errorbar(bx, bm, yerr=bs, fmt="o", color="#ffffff", ms=5, lw=1.4,
                zorder=5, label="decile means ± 2SE")
    xs = np.linspace(tt.min(), tt.max(), 50)
    ax.plot(xs, slope * xs, color="#ef4444", lw=1.6, zorder=6,
            label=f"rung-3 ATE {slope * CONTRAST:+.4f} per +{CONTRAST:.2f} "
                  f"({slope:+.3f}/unit)")
    qb = np.polyfit(tt, yt, 2)
    ax.plot(xs, np.polyval(qb, xs), color="#f59e0b", lw=1.2, ls="--", zorder=6,
            label=f"quadratic (p={quad_p:.2f})")
    ax.set_xlabel("residualized HDI  (T − E[T|W], rung-3's own cross-fitted residuals)")
    ax.set_ylabel("residualized outcome")
    ax.set_title("Robinson residual-on-residual — rung 3's final stage, drawn")
    _legend(ax, fontsize=8.5, ncol=2)
    figs["robinson"] = _svg(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=BG)
    ax = axes[0]; _style(ax)
    yy = np.arange(len(rows))
    ax.barh(yy, [w for *_, w in rows], color="#3b82f6")
    ax.set_yticks(yy, [f"{r} (sd {s_:.3f})" for r, _, s_, _ in rows], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("share of DML weight  (n × Var(T|W))")
    ax.set_title("where the pooled estimate localizes (rung 3)")
    ax = axes[1]; _style(ax)
    xx = np.arange(len(rungs)); wdt = 0.38
    ax.bar(xx - wdt / 2, [r["r2_y"] for r in rungs], wdt, color="#16a34a", label="E[Y|W]")
    ax.bar(xx + wdt / 2, [r["r2_t"] for r in rungs], wdt, color="#7c3aed", label="E[T|W]")
    ax.set_xticks(xx, [r["name"].replace(" — ", "\n") for r in rungs], fontsize=7)
    ax.set_ylabel("out-of-fold R²")
    ax.set_title("first-stage quality of the fitted rungs")
    _legend(ax, fontsize=9)
    figs["weights"] = _svg(fig)

    # ---- html ---------------------------------------------------------------- #
    nrows = "".join(f"<tr><td>{r['name']}</td><td>{r['ate']:+.4f}</td>"
                    f"<td>{r['r2_y']:.4f}</td><td>{r['r2_t']:.4f}</td></tr>"
                    for r in rungs)
    sel_note = (f"<p class=\"sub\">'auto' selected: {r3['selected']}</p>"
                if r3["selected"] else "")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Emerging Destinations &mdash; Model Diagnostics</title>
<style>
  body{{margin:0;background:{BG};color:{FG};font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}}
  header{{padding:24px 28px 6px}} h1{{margin:0 0 6px;font-size:22px}}
  .sub{{color:{MUT};max-width:80ch}} .wrap{{padding:6px 22px 44px}}
  .card{{background:{CARD};border:1px solid #1e2740;border-radius:14px;padding:16px;margin-top:16px;overflow:auto}}
  h2{{font-size:15px;margin:0 0 8px}} td,th{{padding:3px 12px;text-align:right;font-size:13px}}
  th{{color:{MUT}}} td:first-child,th:first-child{{text-align:left}}
  code{{background:#1a2238;border:1px solid #26314e;border-radius:5px;padding:1px 6px;font-size:13px}}
  ul{{margin:6px 0 0;padding-left:20px}} li{{margin:4px 0;font-size:14px}}
  svg{{max-width:100%;height:auto}}
  .zoom{{overflow:hidden;border-radius:8px;cursor:grab;touch-action:none}}
  .zoom:active{{cursor:grabbing}} .zoom svg{{display:block;transform-origin:0 0}}
</style></head><body>
<header><h1>Model diagnostics &mdash; pooled ladder</h1>
<p class="sub">Assumptions and error metrics for the linear (rung 1) and DML (rungs 2&ndash;4)
models on <code>{outcome}</code> (n={n:,}; folds={DML_CV}, seed={SEED}).
<b>Every number comes from the fitted objects <code>estimate.py</code> reports</b> — rung 1 via
the identical <code>adjusted_coef</code> machinery, rungs 2&ndash;4 as the very estimators from
<code>fit_pooled_dml</code>, inspected through econml's own cross-fitted residuals.
Exchangeability is untestable and is probed by <code>refute.py</code>; clustered inference
and LOO fragility are reported by <code>estimate.py</code>. No new pass/fail thresholds here.<br/>
<b>Figures are zoomable:</b> scroll to zoom, drag to pan, double-click to reset.</p></header>
<div class="wrap">
  <div class="card"><h2>Numbers</h2>
    <ul>
      <li>rung 1 OLS (adjusted_coef): &beta; per +{CONTRAST:.2f} = <b>{beta1 * CONTRAST:+.4f}</b>
          (p={p1:.3f}, dof={dof1}); R&sup2; = <b>{r2:.4f}</b>, adj R&sup2; = {adj_r2:.4f}, residual sd = {resid_sd:.4f}</li>
      <li>Robinson slope from rung 3's own residuals = <b>{slope * CONTRAST:+.4f}</b> per +{CONTRAST:.2f}
          ({'matches' if slope_match else 'MISMATCH vs'} the rung-3 reported ATE {r3['ate']:+.4f})</li>
      <li>functional form (rung-3 residual scale): quadratic coefficient {quad_coef:+.3f},
          p = <b>{quad_p:.3f}</b> (HC1) &mdash; the U-shape audit</li>
      <li>identifying variation: sd(HDI) = {sd_t:.4f} &rarr; sd(HDI&thinsp;|&thinsp;W) = <b>{sd_tt:.4f}</b>
          ({sd_tt**2 / sd_t**2:.1%} of treatment variance survives the adjustment set)</li>
    </ul>
    <table><tr><th>fitted rung</th><th>ATE per +{CONTRAST:.2f}</th><th>OOF R&sup2; E[Y|W]</th><th>OOF R&sup2; E[T|W]</th></tr>{nrows}</table>
    {sel_note}
  </div>
  <div class="card"><h2>rung 1 OLS residuals</h2><div class="zoom">{figs['ols']}</div></div>
  <div class="card"><h2>DML final stage (Robinson) — rung 3's own residuals</h2>
    <p class="sub">Each point is a user after partialling out region + year from both axes,
    using the fitted rung-3 estimator's cross-fitted first-stage residuals. The red line's
    slope IS the rung-3 estimate; flat decile means around it = no hidden nonlinearity in
    the effect; the dashed quadratic should hug the line if the single-slope final stage
    is adequate.</p><div class="zoom">{figs['robinson']}</div></div>
  <div class="card"><h2>Self-localization &amp; nuisance quality</h2><div class="zoom">{figs['weights']}</div></div>
</div>
<script>
document.querySelectorAll('.zoom').forEach(z => {{
  const svg = z.querySelector('svg');
  if (!svg) return;
  let s = 1, tx = 0, ty = 0, drag = false, lx = 0, ly = 0;
  const apply = () => {{ svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{s}})`; }};
  z.addEventListener('wheel', e => {{
    e.preventDefault();
    const r = z.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const ns = Math.min(Math.max(s * (e.deltaY < 0 ? 1.2 : 1 / 1.2), 1), 10);
    tx = mx - (mx - tx) * (ns / s); ty = my - (my - ty) * (ns / s); s = ns;
    if (s === 1) {{ tx = 0; ty = 0; }}
    apply();
  }}, {{passive: false}});
  z.addEventListener('mousedown', e => {{
    if (s > 1) {{ drag = true; lx = e.clientX; ly = e.clientY; e.preventDefault(); }}
  }});
  window.addEventListener('mousemove', e => {{
    if (drag) {{ tx += e.clientX - lx; ty += e.clientY - ly; lx = e.clientX; ly = e.clientY; apply(); }}
  }});
  window.addEventListener('mouseup', () => drag = false);
  z.addEventListener('dblclick', () => {{ s = 1; tx = 0; ty = 0; apply(); }});
}});
</script>
</body></html>""", encoding="utf-8")
    print(f"\n-> wrote {OUT}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Pooled-ladder model diagnostics.")
    p.add_argument("--run", action="store_true")
    p.add_argument("--outcome", default="y_first_trip")
    args = p.parse_args()
    if args.run:
        return run(args.outcome)
    print("Nothing to do: pass --run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
