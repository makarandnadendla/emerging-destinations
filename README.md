# Emerging Destinations

Does origin-country development predict how far off the beaten path travelers go inside Georgia? A portfolio-grade causal analysis using geotagged Flickr photos as a natural experiment in tourist self-sorting.

## What's here

- **[`SPEC.md`](SPEC.md)** — the full build spec (analysis design, identification strategy, deliverables).
- **[`CLAUDE.md`](CLAUDE.md)** — project notes and context.
- **`pipeline/extract/`** — extraction scripts (OSM POIs, Flickr photos/users, Nominatim geocoder, indicator panels from World Bank / UNDP / WHO).

## Setup

```powershell
git clone https://github.com/makarandnadendla/emerging-destinations.git
cd emerging-destinations
uv sync
copy .env.example .env       # then fill in keys
```

Requires a Flickr API key (free, auto-approved) and a Backblaze B2 bucket (S3-compatible). See `.env.example` and [SPEC §5](SPEC.md) for credential details.

## Pipeline status

| Stage | Status |
|---|---|
| Extract (Phase 1) | ✅ done — 5 scripts, ~99k photos + ~1.8k users + 38k POIs + 5 country-year indicators |
| Transform (Phase 2, Stage T) | 🚧 in progress |
| Analyze (Phase 3) | ⏳ pending |
| Findings deck | ⏳ pending |
| Methodology PDF | ⏳ pending |

See [SPEC §10](SPEC.md) for the full 2-week timeline.
