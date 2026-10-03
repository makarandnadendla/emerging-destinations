# docs/ — the real-data slide deck

Five-slide horizontal deck presenting the study's real results: twin photo-density
heatmaps of Japan (long-haul vs regional, season-filterable), the within-region
HDI slopes (the H3 finding), region means, and the full methods/results summary.

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
