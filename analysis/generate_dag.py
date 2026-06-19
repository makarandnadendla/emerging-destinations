"""
Generate analysis/dag.html with the causal DAG baked in as STATIC inline SVG.

No JS, no CDN, no build step — the graph is drawn into the HTML so it renders
when the file is opened directly (file://) or served. Layout is hand-placed;
long edges use orthogonal routing so no edge passes through an unrelated node box.

Run:  python analysis/generate_dag.py
"""
from __future__ import annotations
from pathlib import Path

OUT = Path(__file__).resolve().parent / "dag.html"

# ---- palette -------------------------------------------------------------
ROLE = {
    "treat":    "#2563eb",
    "outcome":  "#16a34a",
    "confound": "#dc2626",
    "med":      "#d97706",
    "collider": "#7c3aed",
    "meas":     "#6b7280",
    "neg":      "#0891b2",
}
EDGE = {
    "eff":  "#3b82f6",
    "conf": "#ef4444",
    "med":  "#f59e0b",
    "sel":  "#a78bfa",
    "meas": "#94a3b8",
    "neg":  "#22d3ee",
}

# ---- nodes: cx, cy, w, h, role, label lines ------------------------------
N = {
    "REGION":  (215, 110, 212, 92, "confound", ["Origin region /", "proximity & culture", "(confounder + H3 modifier)"]),
    "YEAR":    (160, 250, 172, 66, "confound", ["Trip year", "(boom, new transport,", "HDI drift)"]),
    "HOMERES": (110, 430, 170, 64, "meas",     ["Home-country", "resolution", "(stated vs modal)"]),
    "HDI":     (320, 430, 162, 66, "treat",    ["Origin development", "(HDI; alt GDPpc,", "LPI, UHC, WGI)"]),
    "OSM":     (560, 275, 180, 58, "meas",     ["OSM mapping", "completeness"]),
    "POI":     (832, 275, 214, 58, "meas",     ["OSM POI density", "(remoteness yardstick)"]),
    "BUDGET":  (620, 500, 172, 52, "med",      ["Trip budget /", "spending power"]),
    "STYLE":   (620, 572, 172, 52, "med",      ["Independent vs", "package travel"]),
    "LENGTH":  (620, 644, 172, 52, "med",      ["Trip length /", "duration"]),
    "CAR":     (620, 716, 172, 52, "med",      ["Self-drive /", "rental-car mobility"]),
    "FLICKR":  (320, 600, 178, 58, "collider", ["Flickr & geotag", "adoption"]),
    "SAMPLE":  (320, 748, 238, 68, "collider", ["IN SAMPLE: traveled", "& geotagged Japan", "[conditioned]"]),
    "HOUR":    (1090, 672, 208, 66, "neg",     ["Hour-of-day", "(negative-control", "outcome)"]),
    "Y":       (1090, 430, 192, 74, "outcome", ["Within-Japan", "remoteness exposure", "(photo-wtd mean", "cell remoteness)"]),
}

def cx(n): return N[n][0]
def cy(n): return N[n][1]

def border_point(node, tx, ty):
    """Point on node's rectangle border along the ray from its center to (tx,ty)."""
    bx, by, w, h = N[node][0], N[node][1], N[node][2], N[node][3]
    dx, dy = tx - bx, ty - by
    if dx == 0 and dy == 0:
        return bx, by
    hw, hh = w / 2.0, h / 2.0
    tx_ = hw / abs(dx) if dx else float("inf")
    ty_ = hh / abs(dy) if dy else float("inf")
    t = min(tx_, ty_)
    return bx + dx * t, by + dy * t

# ---- edges ---------------------------------------------------------------
# kind: straight | quad | poly
#   straight: (src, dst)
#   quad:     (src, dst, (vx, vy))
#   poly:     (list_of_points)   first = src border contact, last = dst border contact
EDGES = [
    ("straight", ("HDI", "Y"),                        "eff",  False, 4.0, None),
    ("straight", ("REGION", "HDI"),                   "conf", False, 2.0, None),
    ("straight", ("YEAR", "HDI"),                     "conf", False, 2.0, None),
    ("poly",     [(215, 64), (215, 34), (1060, 34), (1060, 394)],            "conf", False, 2.0, None),
    ("poly",     [(245, 250), (350, 250), (350, 66), (1085, 66), (1085, 394)], "conf", False, 2.0, None),
    ("poly",     [(320, 110), (460, 110), (460, 748), (438, 748)],           "conf", False, 2.0, None),
    ("straight", ("HDI", "BUDGET"),                   "med",  False, 2.0, None),
    ("quad",     ("HDI", "STYLE", (486, 548)),        "med",  False, 2.0, None),
    ("quad",     ("HDI", "LENGTH", (486, 620)),       "med",  False, 2.0, None),
    ("quad",     ("HDI", "CAR", (470, 706)),          "med",  False, 2.0, None),
    ("straight", ("BUDGET", "Y"),                     "med",  False, 2.0, None),
    ("straight", ("STYLE", "Y"),                      "med",  False, 2.0, None),
    ("straight", ("LENGTH", "Y"),                     "med",  False, 2.0, None),
    ("quad",     ("CAR", "Y", (800, 716)),            "med",  False, 2.0, None),
    ("straight", ("HDI", "FLICKR"),                   "sel",  False, 2.0, None),
    ("straight", ("FLICKR", "SAMPLE"),                "sel",  False, 2.0, None),
    ("straight", ("OSM", "POI"),                      "meas", False, 2.0, None),
    ("straight", ("POI", "Y"),                        "meas", False, 2.0, None),
    ("straight", ("HOMERES", "HDI"),                  "meas", True,  2.0, ("meas. error", 218, 416)),
    ("poly",     [(400, 456), (512, 456), (512, 846), (1090, 846), (1090, 705)], "neg", True, 2.0, ("expect ≈ 0", 720, 862)),
]

def rounded_rect(node):
    bx, by, w, h, role, lines = N[node]
    x, y = bx - w / 2, by - h / 2
    fill = ROLE[role]
    dash = ' stroke-dasharray="7 5"' if role == "collider" else ""
    parts = [
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" rx="11" ry="11" '
        f'fill="{fill}" stroke="#0b0f1a" stroke-width="1.5"{dash}/>'
    ]
    n = len(lines)
    first_y = by - (n - 1) * 8 + 5
    spans = []
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else 16
        weight = 700 if i == 0 else 500
        yy = f' y="{first_y:.0f}"' if i == 0 else f' dy="{dy}"'
        spans.append(f'<tspan x="{bx}"{yy} font-weight="{weight}">{ln}</tspan>')
    parts.append(
        f'<text text-anchor="middle" fill="#ffffff" font-size="12.5" '
        f'font-family="Segoe UI, system-ui, sans-serif">{"".join(spans)}</text>'
    )
    return "\n".join(parts)

def edge_svg(e):
    kind, payload, ckey, dash, width, label = e
    color = EDGE[ckey]
    da = ' stroke-dasharray="6 5"' if dash else ""
    marker = f'arr-{ckey}'
    out = []
    if kind == "straight":
        src, dst = payload
        sx, sy = border_point(src, cx(dst), cy(dst))
        ex, ey = border_point(dst, cx(src), cy(src))
        out.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{da} marker-end="url(#{marker})"/>'
        )
    elif kind == "quad":
        src, dst, via = payload
        vx, vy = via
        sx, sy = border_point(src, vx, vy)
        ex, ey = border_point(dst, vx, vy)
        out.append(
            f'<path d="M {sx:.1f} {sy:.1f} Q {vx} {vy} {ex:.1f} {ey:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="{width}"{da} marker-end="url(#{marker})"/>'
        )
    elif kind == "poly":
        pts = payload
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        out.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da} '
            f'stroke-linejoin="round" marker-end="url(#{marker})"/>'
        )
    if label:
        txt, lx, ly = label
        out.append(
            f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="11.5" fill="#cbd5e1" '
            f'font-family="Segoe UI, system-ui, sans-serif" '
            f'style="paint-order:stroke;stroke:#0b0f1a;stroke-width:3px;stroke-linejoin:round">{txt}</text>'
        )
    return "\n".join(out)

def markers():
    out = []
    for key, color in EDGE.items():
        out.append(
            f'<marker id="arr-{key}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
    return "\n".join(out)

def build_svg():
    W, H = 1240, 900
    body = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, system-ui, sans-serif">']
    body.append(f"<defs>\n{markers()}\n</defs>")
    # effect-of-interest annotation
    body.append('<text x="690" y="416" text-anchor="middle" font-size="12" fill="#93c5fd" '
                'style="paint-order:stroke;stroke:#0b0f1a;stroke-width:3px">effect of interest</text>')
    for e in EDGES:           # edges first (under nodes)
        body.append(edge_svg(e))
    for node in N:            # nodes on top
        body.append(rounded_rect(node))
    body.append("</svg>")
    return "\n".join(body)

LEGEND = [
    ("treat", "Treatment"), ("outcome", "Outcome"), ("confound", "Confounder (block)"),
    ("med", "Mediator (do NOT adjust)"), ("collider", "Selection / collider (conditioned)"),
    ("meas", "Measurement"), ("neg", "Negative control"),
]

def build_html():
    svg = build_svg()
    legend = "".join(
        f'<span><i style="background:{ROLE[k]}"></i>{lbl}</span>' for k, lbl in LEGEND
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Emerging Destinations &mdash; Causal DAG</title>
<style>
  body{{margin:0;background:#0b0f1a;color:#e7ecf5;
       font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
  header{{padding:24px 28px 6px}} h1{{margin:0 0 6px;font-size:22px}}
  .sub{{color:#9aa6bd;max-width:78ch}}
  .wrap{{padding:6px 22px 44px}}
  .card{{background:#121829;border:1px solid #1e2740;border-radius:14px;padding:16px;margin-top:16px;overflow:auto}}
  .legend{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:12px}}
  .legend span{{display:inline-flex;align-items:center;gap:7px;color:#9aa6bd;font-size:13px}}
  .legend i{{width:13px;height:13px;border-radius:3px;display:inline-block}}
  h2{{font-size:15px;margin:0 0 8px}}
  code{{background:#1a2238;border:1px solid #26314e;border-radius:5px;padding:1px 6px;font-size:13px}}
  ul{{margin:6px 0 0;padding-left:20px}} li{{margin:5px 0}} .tag{{font-weight:600}}
</style></head>
<body>
<header>
  <h1>Causal DAG &mdash; origin development &rarr; within-Japan remoteness</h1>
  <p class="sub">Estimand: the effect of an inbound traveler's <b>origin-country development</b>
  (HDI; with GDP&thinsp;pc&thinsp;PPP, LPI, UHC, WGI as alternative operationalizations) on
  <b>how off-the-beaten-path they go inside Japan</b> (photo-weighted mean cell remoteness,
  remoteness = inverse OSM POI density). Thick blue arrow = effect of interest.</p>
</header>
<div class="wrap">
  <div class="card">
    {svg}
    <div class="legend">{legend}</div>
  </div>
  <div class="card">
    <h2>Identification strategy this DAG implies</h2>
    <ul>
      <li><span class="tag">Adjustment set = <code>origin region</code> + <code>trip_year</code>.</span>
          These are the only true back-door confounders; they close
          <code>HDI &larr; REGION &rarr; Y</code> and <code>HDI &larr; YEAR &rarr; Y</code>.</li>
      <li><span class="tag">Region does double duty</span> &mdash; confounder <i>and</i> H3 effect-modifier.
          Estimate <b>within-region HDI slopes (CATE by region)</b> rather than one pooled ATE: that blocks the
          back-door <i>and</i> produces the heterogeneity headline.</li>
      <li><span class="tag">Mediators stay open</span> (budget, independent travel, trip length, self-drive).
          We want the <b>total</b> effect, so we do <b>not</b> condition on them. Corollary: GDPpc / LPI / UHC / WGI
          are <b>alternative treatments</b>, run separately &mdash; never co-adjusted with HDI.</li>
      <li><span class="tag">Selection collider cannot be closed.</span> The Flickr sample is fixed, so
          <code>SAMPLE</code> is conditioned by construction, opening
          <code>HDI &rarr; Flickr &rarr; [SAMPLE] &larr; REGION &rarr; Y</code>. Adjusting <code>region</code>
          blocks its final leg; residual selection is probed by the <b>hour-of-day negative control</b>, an
          <b>E-value</b>, and the <b>stated-vs-modal home-resolution agreement</b> check.</li>
      <li><span class="tag">Design strength:</span> the remoteness yardstick (OSM POI density) is exogenous to
          tourist behavior, and its error (rural under-mapping) is independent of origin HDI &rarr;
          non-differential &rarr; attenuation toward null, not a spurious effect.</li>
    </ul>
    <p class="sub" style="margin-top:10px">Note: the outcome is photo-count weighted; because photo count is a
    descendant of trip length (a mediator) and Flickr engagement, an unweighted per-cell outcome is run as a
    sensitivity spec.</p>
  </div>
</div>
</body></html>
"""

OUT.write_text(build_html(), encoding="utf-8")
print(f"wrote {OUT}")
