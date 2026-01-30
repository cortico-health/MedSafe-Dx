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

def _render_markdown_file(path: str, title: str) -> Response:
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
    </style>
</head>
<body>
    <div class="back-link"><a href="/">&larr; Back to Leaderboard</a></div>
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

@app.get("/")
async def read_index():
    return FileResponse('static/leaderboard.html')

@app.get("/methodology.html")
async def read_methodology():
    return RedirectResponse(url="/report.html", status_code=301)

@app.get("/report.html")
async def read_report():
    try:
        return _render_markdown_file(
            str(PROJECT_ROOT / "BENCHMARK_REPORT.md"),
            "MedSafe-Dx Methodology & Results",
        )
    except FileNotFoundError:
        return Response(content="Report not found. Ensure BENCHMARK_REPORT.md is mounted.", status_code=404)

@app.get("/publish-tables.html")
async def read_publish_tables():
    try:
        return _render_markdown_file(
            "/app/results/analysis/publish_tables.md",
            "MedSafe-Dx Publication Tables",
        )
    except FileNotFoundError:
        return Response(content="Publish tables not found", status_code=404)

@app.get("/case-breakdown.html")
async def read_case_breakdown():
    try:
        return _render_markdown_file(
            "/app/results/analysis/case_breakdown_tables.md",
            "MedSafe-Dx Case Breakdown",
        )
    except FileNotFoundError:
        return Response(content="Case breakdown not found", status_code=404)

@app.get("/README.md")
async def read_readme():
    return FileResponse(str(PROJECT_ROOT / "README.md"), media_type="text/markdown")

@app.get("/results-summary.html")
async def read_results_summary():
    try:
        return _render_markdown_file(
            str(PROJECT_ROOT / "BENCHMARK_REPORT.md"),
            "MedSafe-Dx Methodology & Results",
        )
    except FileNotFoundError:
        return Response(content="Report not found", status_code=404)

app.mount("/", StaticFiles(directory="static"), name="static")
