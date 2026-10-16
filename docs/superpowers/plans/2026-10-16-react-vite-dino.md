# React + Vite Chrome Dino & Laya/Vega Web Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modern React + Vite web application running Chrome Dino at 60 FPS on HTML5 Canvas with a FastAPI backend serving Laya and Vega typed decision models to benchmark latency jitter and 45ms lag compensation.

**Architecture:** A web architecture comprising a FastAPI Python server in `backend/` exposing `/predict` and `/health`, and a React + Vite + TypeScript single-page application in `frontend/` rendering the Dino game on HTML5 canvas with a retro terminal telemetry dashboard, rolling latency sparkline, and client-side 45ms horizon hold scheduling.

**Tech Stack:** React 19 / 18, Vite, TypeScript, HTML5 Canvas 2D, Python 3.12, FastAPI, Uvicorn, `laya`, `vegaml`.

**Spec:** [`docs/superpowers/specs/2026-10-16-react-vite-dino-design.md`](file:///Users/milan/Documents/Desk/others/layadino/docs/superpowers/specs/2026-10-16-react-vite-dino-design.md)

## Global Constraints

- Backend server: FastAPI on `http://localhost:8000` with CORS enabled for `http://localhost:5173`.
- Model runtime: Python `.venv` with `laya>=0.3.6` and `vegaml>=0.9.0`.
- Frontend: React + Vite + TypeScript in `frontend/`, running on `http://localhost:5173`.
- Game loop: `requestAnimationFrame` 60 FPS on HTML5 `<canvas>`.
- Theme: Minimalist Retro Terminal / Arcade (monospace fonts, phosphor accents, CRT styling).
- Hold target: 45.0 ms for Round 2 lag compensation.

---

### Task 1: FastAPI Python Backend Server

**Files:**
- Create: `backend/server.py`
- Create: `backend/test_server.py`

**Interfaces:**
- Produces:
  - `GET /health` -> `{"status": "ok", "models": ["laya", "vega"]}`
  - `POST /predict` -> `{"id": int, "model": str, "action": "jump"|"run", "confidence": float, "inference_ms": float}`

- [ ] **Step 1: Write backend server with CORS and model endpoints**

Create `backend/server.py`:
```python
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

app = fastapi.FastAPI(title="Dino Decision Brain Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_lock = threading.Lock()
laya_model = None
vega_model = None

LAYA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino game. Decide whether to JUMP or RUN to avoid cactus collision.",
        "criteria": {
            "jump": "Approaching cactus within jump trigger distance and Dino is grounded. Leap now.",
            "run": "Runway is clear, cactus is far away, or Dino is already airborne.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Avoid cactus obstacle in Chrome Dino.",
        "criteria": {
            "jump": "Cactus is close and Dino must jump to avoid collision.",
            "run": "Runway is clear or obstacle is far away.",
        },
    }
}


def get_laya():
    global laya_model
    if laya_model is None:
        import laya
        laya_model = laya.load("convaiinnovations/laya")
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import vegaml
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        vega_model = vegaml.load(device=device)
    return vega_model


class PredictRequest(BaseModel):
    id: int = 0
    model: str = "laya"
    state: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok", "models": ["laya", "vega"]}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    model_name = req.model.lower()
    state = req.state

    with model_lock:
        if model_name == "vega":
            v = get_vega()
            dist = state.get("distance_to_cactus", 999.0)
            grounded = state.get("dino_is_grounded", True)
            budget = state.get("latency_budget_ms", 0.0)
            formatted_state = f"Dino approaching cactus at distance {dist:.1f}. Grounded: {grounded}. Latency budget: {budget:.0f}ms."
            res = v.decide(formatted_state, VEGA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("confidence", 0.9))
        else:
            l = get_laya()
            dist = state.get("distance_to_cactus", 999.0)
            grounded = state.get("dino_is_grounded", True)
            speed = state.get("speed", 1.0)
            width = state.get("cactus_width", 3)
            budget = state.get("latency_budget_ms", 0.0)
            formatted_state = {
                "cactus_dist": f"{dist:.1f} cells",
                "grounded": "yes" if grounded else "no",
                "speed": f"{speed:.2f}",
                "width": width,
                "latency_budget": f"{budget:.0f}ms",
            }
            res = l.predict(state=formatted_state, questions=LAYA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("answer_confidence", ans.get("confidence", 0.9)))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "id": req.id,
        "model": model_name,
        "action": action.lower(),
        "confidence": round(conf, 4),
        "inference_ms": round(elapsed_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 2: Write test script for backend API**

Create `backend/test_server.py`:
```python
from fastapi.testclient import TestClient
from backend.server import app

def test_api():
    client = TestClient(app)
    # 1. Health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    # 2. Predict Laya
    req = {
        "id": 1,
        "model": "laya",
        "state": {
            "distance_to_cactus": 8.0,
            "cactus_width": 3,
            "dino_is_grounded": True,
            "speed": 1.0,
            "latency_budget_ms": 0.0,
        },
    }
    res = client.post("/predict", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == 1
    assert data["action"] in ["jump", "run"]
    assert data["inference_ms"] > 0
    print(f"Backend API test passed! Response: {data}")

if __name__ == "__main__":
    test_api()
```

- [ ] **Step 3: Run the test script**

Run: `.venv/bin/python backend/test_server.py`
Expected: PASS ("Backend API test passed!")

- [ ] **Step 4: Commit**

```bash
git add backend/
git commit -m "feat(backend): implement FastAPI inference server for Laya and Vega"
```

---

### Task 2: React + Vite Frontend Scaffolding

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/tsconfig.json`

- [ ] **Step 1: Create package.json and configuration files**

Create `frontend/package.json`:
```json
{
  "name": "layadino-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "lucide-react": "^0.453.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "typescript": "^5.6.3",
    "vite": "^5.4.10"
  }
}
```

Create `frontend/vite.config.ts`:
```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
});
```

Create `frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LAYADINO // Chrome Dino Laya & Vega Benchmark</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap" rel="stylesheet">
  </head>
  <body class="bg-black text-white select-none">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Create `frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 2: Install dependencies**

Run: `cd frontend && npm install`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/vite.config.ts frontend/index.html frontend/tsconfig.json
git commit -m "feat(frontend): scaffold React + Vite web application"
```

---

### Task 3: Canvas Dino Game Engine & Pixel Art Physics

**Files:**
- Create: `frontend/src/game/types.ts`
- Create: `frontend/src/game/sprites.ts`
- Create: `frontend/src/game/engine.ts`

- [ ] **Step 1: Define game types**

Create `frontend/src/game/types.ts`:
```ts
export type DriverMode = 'human' | 'laya' | 'vega';
export type ExperimentRound = 'raw_async' | 'fixed_45ms_hold';

export interface Obstacle {
  x: number;
  width: number;
  height: number;
  type: 'small' | 'large' | 'cluster';
}

export interface DinoObservation {
  distance_to_cactus: number;
  cactus_width: number;
  cactus_height: number;
  dino_is_grounded: boolean;
  speed: number;
  latency_budget_ms: number;
}
```

- [ ] **Step 2: Implement pixel art drawing routines**

Create `frontend/src/game/sprites.ts`:
Renders retro pixel art Dino (running frame 1, frame 2, jump, death) and cacti onto HTML5 canvas with sharp pixelated rendering.

- [ ] **Step 3: Implement GameEngine**

Create `frontend/src/game/engine.ts`:
Manages physics (speed scaling, gravity, jump arc, cacti spawning, collision detection, score) running on `requestAnimationFrame`.

---

### Task 4: API Client & Lag Compensation Agent Store

**Files:**
- Create: `frontend/src/telemetry/api.ts`
- Create: `frontend/src/telemetry/stats.ts`

- [ ] **Step 1: Implement fetch API client**

Create `frontend/src/telemetry/api.ts`:
Sends `POST http://localhost:8000/predict` with timeout and returns predicted action, confidence, and latency.

- [ ] **Step 2: Implement Rolling Statistics & Percentiles**

Create `frontend/src/telemetry/stats.ts`:
Computes rolling p50, p90, p99, jitter ($\sigma$), and deadline misses.

---

### Task 5: Retro Terminal Dashboard & Latency Sparkline UI

**Files:**
- Create: `frontend/src/telemetry/Sparkline.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/main.tsx`

- [ ] **Step 1: Implement Sparkline component**
- [ ] **Step 2: Implement full App dashboard with CRT styling and hotkeys**
- [ ] **Step 3: Verify build (`npm run build`)**

---

### Task 6: End-to-End Verification

- [ ] **Step 1: Start backend on port 8000**
- [ ] **Step 2: Start frontend on port 5173**
- [ ] **Step 3: Verify gameplay in human mode and AI autopilot with Laya and Vega**
