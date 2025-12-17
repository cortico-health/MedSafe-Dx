import json
import os
import glob
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

LEADERBOARD_DIR = '/app/leaderboard'

def calculate_safety_score(safety: Dict[str, int]) -> int:
    return safety.get('missed_escalations', 0) + safety.get('overconfident_wrong', 0) + safety.get('unsafe_reassurance', 0)

def get_leaderboard_data() -> List[Dict[str, Any]]:
    results = []
    if os.path.exists(LEADERBOARD_DIR):
        for json_file in glob.glob(os.path.join(LEADERBOARD_DIR, '*.json')):
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    # Only include files that look like evaluation results (must have 'safety' key)
                    if 'safety' in result:
                        results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

        # Sort results by safety score (ascending), then by top-3 recall (descending)
        results.sort(key=lambda x: (
            calculate_safety_score(x.get('safety', {})),
            x.get('safety', {}).get('missed_escalations', 0),
            -(x.get('effectiveness', {}).get('top3_recall') or 0)
        ))
    return results

@app.get("/leaderboard-data.json")
async def leaderboard_data():
    return JSONResponse(content=get_leaderboard_data())

@app.get("/")
async def read_index():
    return FileResponse('static/leaderboard.html')

@app.get("/methodology.html")
async def read_methodology():
    return FileResponse('static/methodology.html')

@app.get("/README.md")
async def read_readme():
    return FileResponse('README.md', media_type='text/markdown')

app.mount("/", StaticFiles(directory="static"), name="static")
