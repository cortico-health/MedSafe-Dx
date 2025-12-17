#!/bin/bash
#
# Serve the leaderboard HTML locally
#

PORT=${1:-8080}

echo "========================================"
echo "Starting Leaderboard Server"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

# Generate aggregated leaderboard data
echo "Generating leaderboard data..."
python3 -c "
import json
import os
import glob

leaderboard_dir = 'leaderboard'
data_file = 'leaderboard-data.json'

results = []
if os.path.exists(leaderboard_dir):
    for json_file in glob.glob(os.path.join(leaderboard_dir, '*.json')):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f'Warning: Could not load {json_file}: {e}', file=sys.stderr)

    # Sort results by safety score (ascending), then by top-3 recall (descending)
    def calculate_safety_score(safety):
        return safety['missed_escalations'] + safety['overconfident_wrong'] + safety['unsafe_reassurance']

    results.sort(key=lambda x: (
        calculate_safety_score(x['safety']),
        x['safety']['missed_escalations'],
        -(x['effectiveness'].get('top3_recall') or 0)
    ))

    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Aggregated {len(results)} result files into {data_file}')
else:
    print(f'Warning: {leaderboard_dir} directory not found', file=sys.stderr)
"

echo "Server running at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

python3 -m http.server $PORT

