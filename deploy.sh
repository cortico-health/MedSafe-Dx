#!/usr/bin/env bash
# Deploy the leaderboard web container.
#
# The Dockerfile bakes web/static into the image, so a page edit is only live
# after a rebuild. This script pulls, rebuilds, restarts, and checks the site.
#
#   ./deploy.sh              # pull main, rebuild, restart, health-check
#   ./deploy.sh --no-pull    # deploy the working tree as-is
set -euo pipefail
cd "$(dirname "$0")"

PORT="${WEB_PORT:-18080}"
PULL=1
[[ "${1:-}" == "--no-pull" ]] && PULL=0

if (( PULL )); then
  echo "==> git pull"
  git pull --ff-only
fi

echo "==> build + restart web ($(git rev-parse --short HEAD))"
docker compose up -d --build web
docker image prune -f >/dev/null

echo "==> health check http://127.0.0.1:${PORT}/"
for _ in $(seq 1 20); do
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/" \
     && curl -sf "http://127.0.0.1:${PORT}/leaderboard-data.json" | grep -q '"model"'; then
    echo "OK: leaderboard up, $(curl -s "http://127.0.0.1:${PORT}/leaderboard-data.json" | grep -o '"model"' | wc -l) rows"
    exit 0
  fi
  sleep 1
done

echo "FAIL: site not responding on port ${PORT}. Last log lines:" >&2
docker compose logs --tail=30 web >&2
exit 1
