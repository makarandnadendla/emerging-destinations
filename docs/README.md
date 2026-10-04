# docs/ — the real-data slide deck

Eight-slide horizontal deck telling the study as a story in three acts:

1. **Hero** — the question.
2. **Act I · Hypotheses** — the four pre-registered rival theories (H1 acclimation,
   H2 status-good, H3 heterogeneous, H4 null), called before unblinding.
3. **Act II · Explore** — a full-bleed photo-density heatmap of Japan,
   season-filterable, with checkbox chips choosing whose photos it shows:
   Pooled (everyone, the default) or any combination of the eight origin
   regions (Europe, North America, East Asia, Southeast Asia, Oceania,
   Latin America, Mid-East & Africa, South/Central Asia); pooled and the
   regions auto-untick each other so counts never double…
4. …and mean off-path-ness by origin region — ending on the trap: those bars
   are confounded, not causal.
5. **Act III · Identify** — the causal map (simplified DAG): why region + year
   are adjusted and mediators are left open.
6. **Act III · Estimate** — the adjusted within-region HDI slopes: the H3 answer.
7. **Act III · Pool it** — the counterfactual analysis choice: one pooled slope
   (the full OLS→DML ladder, from the recorded runs) lands on a null — showing
   how opposite-signed regional effects cancel when you don't estimate
   per region.
8. **Appendix** — full methods & results.

This is the real-data successor to the gitignored `prototype/` mock deck —
same visual design, every number real.

## Data provenance (nothing hand-typed)

`data/aggregates.js` is **generated** by `analysis/export_aggregates.py`:

- model numbers (gate, within-region slopes + LOO, pooled ladder) are read from
  `analysis/outputs/estimate_results.json`, written by
  `uv run python analysis/estimate.py --run`;
- refutation rows are read from the reports written by
  `uv run python analysis/refute.py --run`;
- data aggregates (cell × group × season photo counts, origin rollups)
  are grouped straight from the DuckDB warehouse — **no
  user-level rows**, with the `min_origin_users` suppression floor applied.

To regenerate after a new run:

```bash
uv run python analysis/estimate.py --run
uv run python analysis/export_aggregates.py
```

## Serving locally

```powershell
python -m http.server 8912 --directory docs
```

then open <http://localhost:8912>. (Also works on GitHub Pages — it's a static
site with **zero external dependencies**: libraries are vendored in `vendor/`
and the maps draw sea + the hand-simplified Japan outline locally, no tile
server. It renders offline.)

**Controls:** `←`/`→` or the dots to navigate, `Space` to pause auto-advance,
swipe on mobile. The Season dropdown on slide 2 filters both maps live.
