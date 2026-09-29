import json
import os
import glob
from pathlib import Path
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
import markdown

app = FastAPI()

# Disable caching middleware for development
class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

LEADERBOARD_DIR = '/app/leaderboard'
APP_ROOT = Path("/app")
PROJECT_ROOT = Path("/app/project")
_CASE_DENOM_CACHE: dict[str, dict[str, int]] = {}


def _load_case_denominators(cases_path: str) -> dict[str, int] | None:
    """
    Returns denominators derived from the frozen test set:
      - cases_expected
      - escalation_required_cases
      - nonurgent_cases
      - ambiguity_acceptable_cases
    """
    if not cases_path:
        return None
    if cases_path in _CASE_DENOM_CACHE:
        return _CASE_DENOM_CACHE[cases_path]

    try:
        p = (APP_ROOT / cases_path).resolve()
        if not str(p).startswith(str(APP_ROOT) + os.sep):
            return None
        with open(p, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        return None

    n = len(cases)
    n_req = sum(1 for c in cases if bool(c.get("escalation_required")))
    n_non = n - n_req
    n_amb = sum(1 for c in cases if bool(c.get("uncertainty_acceptable")))

    denoms = {
        "cases_expected": n,
        "escalation_required_cases": n_req,
        "nonurgent_cases": n_non,
        "ambiguity_acceptable_cases": n_amb,
    }
    _CASE_DENOM_CACHE[cases_path] = denoms
    return denoms


def get_leaderboard_data() -> List[Dict[str, Any]]:
    results = []
    if os.path.exists(LEADERBOARD_DIR):
        # Only load evaluation files (ending in -eval.json)
        for json_file in glob.glob(os.path.join(LEADERBOARD_DIR, '*-eval.json')):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    # Attach denominators + derived, publication-friendly rates when possible.
                    # This avoids requiring eval JSON regeneration just to show correct denominators.
                    cases_path = result.get("cases_path")
                    denoms = result.get("denominators") or _load_case_denominators(cases_path)
                    if denoms:
                        result["denominators"] = denoms
                        eff = result.get("effectiveness") or {}
                        over = eff.get("over_escalation")
                        if over is None:
                            over = (result.get("informational") or {}).get("overdiagnosis") or 0
                        nonurgent = denoms.get("nonurgent_cases") or 0
                        eff["over_escalation_rate_nonurgent"] = (
                            (float(over) / float(nonurgent)) if nonurgent else None
                        )
                        result["effectiveness"] = eff
                    results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

        def sort_key(x: Dict[str, Any]):
            safety_pass_rate = x.get("safety_pass_rate")
            if safety_pass_rate is None:
                safety_pass_rate = -1.0

            safety = x.get("safety") or {}
            missed_escalations = float(safety.get("missed_escalations") or 0)

            effectiveness = x.get("effectiveness") or {}
            over_escalation_rate = effectiveness.get("over_escalation_rate")
            if over_escalation_rate is None:
                over_escalation = (
                    effectiveness.get("over_escalation")
                    or (x.get("informational") or {}).get("overdiagnosis")
                    or 0
                )
                cases = x.get("cases_expected") or x.get("cases") or 0
                over_escalation_rate = (float(over_escalation) / float(cases)) if cases else 1.0

            top3_recall = float(effectiveness.get("top3_recall") or 0)

            return (
                -float(safety_pass_rate),
                missed_escalations,
                float(over_escalation_rate),
                -top3_recall,
            )

        # Sort results by safety pass rate (descending), then tie-break.
        results.sort(key=sort_key)
    return results

def _render_markdown_file(path: str, title: str, extra_links: str = "") -> Response:
    """Render a markdown file as a page. extra_links is HTML we append to the back-link row."""
    p = Path(path)
    md_content = p.read_text(encoding="utf-8")

    try:
        st = p.stat()
        mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        size_bytes = st.st_size
    except Exception:
        mtime_utc = "unknown"
        size_bytes = -1

    sha = hashlib.sha256(md_content.encode("utf-8")).hexdigest()[:12]

    body_html = markdown.markdown(
        md_content,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )
    # Wide markdown tables scroll inside their own box on narrow screens
    # instead of forcing the whole page to scroll sideways.
    body_html = body_html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        h1, h2, h3 {{ color: #4b54f6; }}
        h1 {{ border-bottom: 2px solid #4b54f6; padding-bottom: 0.5rem; }}
        h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
        th {{ background: #f8f9fa; }}
        tr:nth-child(even) {{ background: #fafafa; }}
        code {{ background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; }}
        a {{ color: #4b54f6; }}
        .back-link {{ margin-bottom: 1rem; }}
        .render-meta {{ margin-top: 2rem; font-size: 0.85rem; color: #666; }}
        img {{ max-width: 100%; height: auto; }}
        a, code {{ overflow-wrap: anywhere; }}
        .table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 1rem 0; }}
        .table-wrap table {{ margin: 0; }}
        @media (max-width: 768px) {{
            body {{ padding: 1rem; font-size: 0.95rem; }}
            h1 {{ font-size: 1.5rem; }}
            h2 {{ font-size: 1.25rem; }}
            th, td {{ padding: 0.35rem 0.5rem; font-size: 0.85rem; white-space: nowrap; }}
            pre {{ padding: 0.75rem; font-size: 0.8rem; }}
        }}
        /* Print: A4, light palette, tables that repeat headers and never clip. */
        @page {{ size: A4; margin: 15mm; }}
        @media print {{
            body {{ max-width: none; padding: 0; color: #000; background: #fff; font-size: 10.5pt; }}
            .back-link {{ display: none; }}
            a {{ color: #000; text-decoration: none; }}
            h1, h2, h3 {{ color: #000; page-break-after: avoid; }}
            table {{ font-size: 9pt; page-break-inside: auto; }}
            thead {{ display: table-header-group; }}
            tr {{ page-break-inside: avoid; }}
            th {{ background: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            tr:nth-child(even) {{ background: #f6f6f6 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
            pre, img {{ page-break-inside: avoid; max-width: 100%; }}
            img {{ max-height: 105mm; width: auto; display: block; margin: 0.5rem auto; }}
            th, td {{ padding: 0.2rem 0.35rem; }}
            table {{ font-size: 8pt; }}
            .render-meta {{ font-size: 8pt; page-break-before: avoid; margin-top: 1rem; }}
            p, li {{ orphans: 3; widows: 3; }}
        }}
    </style>
</head>
<body>
    <div class="back-link"><a href="/">&larr; Back to Leaderboard</a>{extra_links}</div>
    <div id="content">{body_html}</div>
    <div class="render-meta">
        Rendered from <code>{p}</code> (mtime UTC: <code>{mtime_utc}</code>, bytes: <code>{size_bytes}</code>, sha256: <code>{sha}</code>)
    </div>
</body>
</html>"""
    return Response(content=html, media_type="text/html")


@app.get("/leaderboard-data.json")
async def leaderboard_data():
    return JSONResponse(content=get_leaderboard_data())

@app.get("/triage-scores.json")
async def triage_scores():
    # Built by scripts/build_triage_board.py; the board ranks by the triage score in it.
    p = os.path.join(LEADERBOARD_DIR, "triage-scores.json")
    if not os.path.exists(p):
        return JSONResponse(content={"error": "triage-scores.json not built"}, status_code=404)
    return FileResponse(p, media_type="application/json")

@app.get("/v02-scores.json")
async def v02_scores():
    # Built by `python3 -m evaluator.v02_score`. Until the paid run exists we serve the
    # SYNTHETIC preview, which carries provenance.any_synthetic so the page labels it.
    for name in ("v02-scores.json", "v02-scores.SYNTHETIC-preview.json"):
        p = os.path.join(LEADERBOARD_DIR, name)
        if os.path.exists(p):
            return FileResponse(p, media_type="application/json")
    return JSONResponse(content={"error": "v0.2 scores not built"}, status_code=404)

@app.get("/")
async def read_index():
    return FileResponse('static/leaderboard.html')

@app.get("/methodology.html")
async def read_methodology():
    return RedirectResponse(url="/report.html", status_code=301)

@app.get("/leaderboard-v02.html")
async def read_leaderboard_v02_archived():
    # The v0.2 preview moved into the frozen archive; this keeps the old URL live.
    return RedirectResponse(url="/archive/v0.2-preview/leaderboard-v02.html", status_code=301)

# The live methodology page is v0.3. The v0 report the preprint cites is frozen at
# /archive/v0.1-preprint/report.html, and the page header links to it.
V03_REPORT_LINKS = (
    ' &middot; <a href="/archive/v0.1-preprint/report.html">Preprint report (v0, archived)</a>'
    ' &middot; <a href="https://doi.org/10.64898/2026.04.14.26350711">medRxiv preprint</a>'
)

@app.get("/report.html")
async def read_report():
    try:
        return _render_markdown_file(
            str(PROJECT_ROOT / "docs" / "METHODOLOGY-v0.3.md"),
            "MedSafe-Dx v0.3: Methodology & Results",
            extra_links=V03_REPORT_LINKS,
        )
    except FileNotFoundError:
        return Response(content="Report not found. Ensure docs/METHODOLOGY-v0.3.md is mounted.", status_code=404)

@app.get("/report-v03.html")
async def read_report_v03():
    # The v0.3 preview URL, kept so shared links still land on the methodology page.
    return RedirectResponse(url="/report.html", status_code=301)

@app.get("/results-summary.html")
async def read_results_summary():
    # Old name for the methodology page.
    return RedirectResponse(url="/report.html", status_code=301)

# /publish-tables.html and /case-breakdown.html are gone: they post-date the preprint
# (added 2026-04-22) and described the v0 250-case run the v0.3 page replaces.

@app.get("/findings-2026-09.html")
async def read_findings_2026_09():
    # The September findings are archived; the frozen copy lives with its board.
    return RedirectResponse(url="/archive/2026-09-refresh/findings-2026-09.html", status_code=301)

@app.get("/README.md")
async def read_readme():
    return FileResponse(str(PROJECT_ROOT / "README.md"), media_type="text/markdown")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
