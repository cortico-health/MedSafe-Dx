#!/usr/bin/env python3
"""Render one markdown file to the static HTML the archive keeps.

web/main.py's `_render_markdown_file` renders a *live* file on every request,
so the HTML changes whenever the underlying markdown does. An archive page
must not do that: once we freeze a commit, the rendered HTML has to stay the
same forever, even after the source file changes or the commit is gone from
disk. This script runs `_render_markdown_file`'s exact markdown-to-HTML
recipe (same extensions, same table-wrap, same CSS) once, against either a
frozen git commit or the current working tree, and writes the result to a
file instead of serving it.

Usage:
    # From a frozen commit (content comes from `git show <commit>:<path>`):
    python3 scripts/web/archive_snapshot.py \\
        --commit 88697e7ac84056089ee8ae3a250e76e6677325c5 \\
        --path BENCHMARK_REPORT.md \\
        --title "MedSafe-Dx (v0): Methodology & Results" \\
        --out web/static/archive/v0.1-preprint/report.html \\
        --banner "the v0 methodology report, as it read when the medRxiv preprint posted" \\
        --banner-date 2026-01-30 (commit 88697e7, no repo changes between it and the 2026-04-22 preprint alignment commit)

    # From the current working tree (no --commit):
    python3 scripts/web/archive_snapshot.py \\
        --path docs/FINDINGS-2026-09.md \\
        --title "MedSafe-Dx: findings from the 2026-09 leaderboard refresh" \\
        --out web/static/archive/2026-09-refresh/findings-2026-09.html \\
        --banner "the September 2026 findings write-up" \\
        --banner-date 2026-09-12
"""
import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same markdown extensions and table-wrap as web/main.py's _render_markdown_file.
MD_EXTENSIONS = ["extra", "tables", "fenced_code", "sane_lists", "toc"]

# Same CSS as web/main.py's _render_markdown_file (kept byte-for-byte so an
# archived page looks like the live report did).
PAGE_CSS = """
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }
        h1, h2, h3 { color: #4b54f6; }
        h1 { border-bottom: 2px solid #4b54f6; padding-bottom: 0.5rem; }
        h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
        th, td { border: 1px solid #ddd; padding: 0.5rem; text-align: left; }
        th { background: #f8f9fa; }
        tr:nth-child(even) { background: #fafafa; }
        code { background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; }
        pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; }
        a { color: #4b54f6; }
        .back-link { margin-bottom: 1rem; }
        .render-meta { margin-top: 2rem; font-size: 0.85rem; color: #666; }
        img { max-width: 100%; height: auto; }
        a, code { overflow-wrap: anywhere; }
        .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1rem 0; }
        .table-wrap table { margin: 0; }
        @media (max-width: 768px) {
            body { padding: 1rem; font-size: 0.95rem; }
            h1 { font-size: 1.5rem; }
            h2 { font-size: 1.25rem; }
            th, td { padding: 0.35rem 0.5rem; font-size: 0.85rem; white-space: nowrap; }
            pre { padding: 0.75rem; font-size: 0.8rem; }
        }
        .archived-banner {
            background: #fef3c7; border-bottom: 1px solid #f5c27a; color: #7a4a00;
            font-size: 0.92rem; padding: 0.7rem 1.2rem; text-align: center; margin: -2rem -2rem 1.5rem;
        }
        .archived-banner a { color: #7a4a00; font-weight: 600; }
        @media (max-width: 768px) { .archived-banner { margin: -1rem -1rem 1rem; } }
"""


def _read_source(commit: str | None, path: str) -> str:
    if commit:
        out = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPO_ROOT, capture_output=True, check=True, text=True,
        )
        return out.stdout
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _commit_date(commit: str) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%cd", "--date=short", commit],
        cwd=REPO_ROOT, capture_output=True, check=True, text=True,
    )
    return out.stdout.strip()


def render(md_content: str, title: str, source_note: str, banner_html: str) -> str:
    body_html = markdown.markdown(md_content, extensions=MD_EXTENSIONS, output_format="html5")
    body_html = body_html.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )
    sha = hashlib.sha256(md_content.encode("utf-8")).hexdigest()[:12]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>{PAGE_CSS}</style>
</head>
<body>
    {banner_html}
    <div class="back-link"><a href="/">&larr; Back to Leaderboard</a></div>
    <div id="content">{body_html}</div>
    <div class="render-meta">
        Archived snapshot. {source_note} sha256: <code>{sha}</code>
    </div>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", default=None, help="Frozen commit to read --path from (git show). Omit to use the working tree.")
    ap.add_argument("--path", required=True, help="Repo-relative path to the markdown file.")
    ap.add_argument("--title", required=True, help="<title> and back-link page title.")
    ap.add_argument("--out", required=True, help="Repo-relative output HTML path.")
    ap.add_argument("--banner", required=True, help="What this page is, for the archived-banner sentence.")
    ap.add_argument("--banner-date", required=True, help="Date shown in the archived-banner sentence (YYYY-MM-DD).")
    args = ap.parse_args()

    md_content = _read_source(args.commit, args.path)

    if args.commit:
        cdate = _commit_date(args.commit)
        source_note = (
            f"Rendered from <code>{args.path}</code> at commit "
            f"<code>{args.commit[:12]}</code> ({cdate})."
        )
    else:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        source_note = f"Rendered from <code>{args.path}</code> (working tree, snapshotted {now})."

    banner_html = (
        '<div class="archived-banner">Archived: '
        f"{args.banner}, {args.banner_date}. Superseded by the revised MedSafe-Dx methodology "
        '(v0.3, in validation). <a href="/">See the current site</a>.</div>'
    )

    html = render(md_content, args.title, source_note, banner_html)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
