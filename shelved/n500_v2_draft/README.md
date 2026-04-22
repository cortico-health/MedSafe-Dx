# Shelved: N=500 / 7-model paper draft

This directory contains a draft manuscript that was **superseded and shelved**. It is preserved for reference but is **not** the published version of MedSafe-Dx (v0).

## Why it's here

The published preprint on medRxiv ([doi.org/10.64898/2026.04.14.26350711](https://doi.org/10.64898/2026.04.14.26350711)) reports the **N=250, 11-model** evaluation that matches the live leaderboard at <https://msdx.cortico.health/> and the methodology in [`../../BENCHMARK_REPORT.md`](../../BENCHMARK_REPORT.md).

This draft (`paper.tex` / `paper.docx` / `paper.pdf`) describes a **different evaluation**:

- **N=500** instead of 250
- **7 models** instead of 11 (GPT-4o, Claude Haiku 4.5, Claude Sonnet 4.6, Claude Opus 4.6, GPT-5 Mini, Gemini 3.1 Pro Preview, GPT-5.4 Pro)
- **Different leaderboard**: GPT-4o ranks #1 at 95.4% Safety Pass Rate
- **Different non-urgent denominator** (194 vs 94)

The associated 500-case prediction artifacts remain in `../../results/artifacts/*-500cases*.json` for reference but are not promoted to the leaderboard.

## Status

Shelved. No further work planned. The published v0 preprint and the live leaderboard are the canonical statements of the benchmark.

## Contents

- `paper.tex`, `paper.bib`, `paper.docx`, `paper.pdf` — manuscript and bibliography
- `Makefile` — LaTeX build target (paths reference this directory)
- `run_frontier_500.sh` — script that produced the 500-case predictions
- LaTeX build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`) — regenerable; kept so the directory is self-contained
