#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================="
echo "   LAYA-VEGA-BENCHMARK // Launcher               "
echo "================================================="

# Check virtual environment
if [ ! -d ".venv" ]; then
  echo "[!] .venv not found. Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r backend/requirements.txt
fi

# 1. Start Python Backend
echo "[*] Starting FastAPI Backend on http://localhost:8000..."
.venv/bin/uvicorn backend.server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap Ctrl+C to kill both servers cleanly
cleanup() {
  echo ""
  echo "[*] Shutting down servers..."
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Start Frontend
echo "[*] Starting Vite React App on http://localhost:5173..."
cd frontend
if [ ! -d "node_modules" ]; then
  echo "[!] Installing frontend dependencies..."
  npm install
fi
npm run dev
