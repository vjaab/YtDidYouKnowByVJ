# Decorator-flow template — "code editor" carousel style

A drop-in replacement for the plain white slide template, built on the same
stack you're already using (Jinja2-style HTML/CSS → Playwright screenshot).

## Why this look instead of a literal copy of the reference infographic

The reference image uses solid pastel blocks + stock-photo illustrations —
great for a school marketing account, but generic for a Python/dev-education
brand. Instead this reskins the *same structure* (numbered blocks, connecting
arrows, icons, colour-coded steps) into a "VS Code / terminal" visual
language — One Dark Pro–inspired accent palette, a terminal chrome bar,
JetBrains Mono for code bits, and Lucide icons picked to match each concept
(function-square, layers, zap, etc). It reads as colourful and dynamic like
the reference, but it's unmistakably *your* brand rather than a generic
infographic template. If you'd rather have literal solid-color panels closer
to the reference, that's a quick variant — just say the word.

## Setup

```bash
npm install lucide-static @fontsource/jetbrains-mono @fontsource/space-grotesk @fontsource/inter
pip install jinja2 playwright --break-system-packages
playwright install chromium
```

`assets/fonts/` and `assets/icons/` in this folder already contain the exact
files `generate.py` expects (copied out of the npm packages above), so you
can use them as-is without reinstalling anything.

## Files

- `templates/decorator_flow.html.j2` — the Jinja2 template. Colour, type,
  spacing all live here as CSS custom properties (`--accent` per step).
- `generate.py` — defines the step content as a plain Python list of dicts
  (`accent`, `icon`, `title`, `desc`), renders the template, screenshots it
  at 1080×1350 (IG 4:5) with `device_scale_factor=2` for a crisp 2160×2700 export.
- `assets/fonts/` — self-hosted woff2 files (embedded as base64 data URIs at
  render time, so the output HTML is fully self-contained — no relative-path
  issues in CI).
- `assets/icons/` — Lucide SVGs (MIT/ISC licensed, safe to ship), inlined so
  `stroke="currentColor"` can be swapped per-card via `--accent`.

## Wiring into your real pipeline

Swap the hardcoded `STEPS` list in `generate.py` for whatever your content
JSON already produces — each entry just needs `title`, `desc`, and an
`accent` + `icon` pair. I'd suggest keeping a small fixed rotation of 5–6
accent/icon pairs (matching your existing `chapters` schema) rather than
generating them per-topic, so the palette stays consistent across posts and
still reads as "your" template at a glance.

For the code-snippet slide type (like your "What is Python Decorators?"
slide), the same chrome bar + hero + footer shell works — the body just
becomes a syntax-highlighted code block instead of the step list. Happy to
build that variant too if useful.
