"""Run the leaderboard web app locally, no docker.

main.py hardcodes the container's /app paths; this points them at the repo.

    uv venv .venv && uv pip install --python .venv/bin/python -r web/requirements.txt
    .venv/bin/python web/dev_serve.py            # http://127.0.0.1:18081
"""
import os
import sys
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
os.chdir(WEB_DIR)          # StaticFiles(directory="static") is cwd-relative
sys.path.insert(0, str(WEB_DIR))

import main  # noqa: E402

main.LEADERBOARD_DIR = str(ROOT / "leaderboard")
main.APP_ROOT = ROOT
main.PROJECT_ROOT = ROOT

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "18081"))
    uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="warning")
