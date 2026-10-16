# React + Vite Chrome Dino & Laya/Vega Web Benchmark
**Design Specification**

*Date: 2026-10-16*  
*Status: Approved*

---

## 1. Overview & Objectives

This project builds a responsive web application running Chrome Dino with a **React + Vite** frontend and a **FastAPI Python** backend. The application serves two typed decision models—**Laya** (Transformer-based) and **Vega** (Symplectic integrator)—to benchmark reactive AI decision-making under network/inference latency and jitter.

### Key Features
1. **Playable Human Game**: A classic Chrome Dino arcade runner rendered on HTML5 `<canvas>` at 60 FPS with responsive keyboard (`Space` / `Up`), touch, and click controls.
2. **AI Autopilot Drivers**: Switch seamlessly between `Human`, `Laya (0.8B)`, and `Vega (0.8B)`.
3. **Experiment Round 1 (Raw Async)**: Dispatches `POST /predict` when an obstacle enters the trigger horizon; executes the model's action immediately upon HTTP response arrival to observe the impact of latency jitter on jump timing.
4. **Experiment Round 2 (Fixed 45ms Horizon Hold / Lag Compensated)**:
   - Projects obstacle position forward by $+45\text{ms}$ ($d_{\text{projected}} = d_{\text{current}} - v \cdot 0.045$) and informs the model via `latency_budget_ms: 45.0`.
   - Holds early decisions in a client-side timer queue until exactly $45\text{ms}$ from dispatch, landing the jump on the exact deterministic target frame.
   - Compares latency variance (p50, p90, p99, and jitter $\sigma$) between Laya and Vega.
5. **Minimalist Retro Terminal/Arcade Theme**: CRT-inspired monospace aesthetic (`JetBrains Mono` / monospace, glowing phosphor accents, live latency sparkline, rolling percentile telemetry).

---

## 2. System Architecture

```
+------------------------------------------------------------------------------------+
|                         FRONTEND (React 19 + Vite + TypeScript)                    |
|                                                                                    |
|  +------------------------------------------------------------------------------+  |
|  |                         Dino Canvas Game Engine (60 FPS)                     |  |
|  |  - High-DPI Canvas Rendering (Retina crisp scaling)                          |  |
|  |  - Physics (gravity, floaty jump arc, obstacle collision)                   |  |
|  |  - Input Listener (Space / Up / Click / Touch)                              |  |
|  |  - Scheduled Action Queue (holds early responses until 45ms)                 |  |
|  +------------------------------------------------------------------------------+  |
|         |                                                                ^         |
|     Observation JSON                                              Action Response  |
|         v                                                                |         |
|  +------------------------------------------------------------------------------+  |
|  |                         Agent & Telemetry Store                              |  |
|  |  - Dispatches fetch('http://localhost:8000/predict')                         |  |
|  |  - Computes rolling p50, p90, p99, and standard deviation (sigma)            |  |
|  |  - Manages controller mode (Human / Laya / Vega)                             |  |
|  +------------------------------------------------------------------------------+  |
+----------------------------------------|-------------------------------------------+
                                         | HTTP POST /predict (JSON)
                                         v
+------------------------------------------------------------------------------------+
|                             BACKEND (FastAPI + Uvicorn)                            |
|                                                                                    |
|  - GET  /health   -> { status: "ok", models: ["laya", "vega"] }                    |
|  - POST /predict  -> { action: "jump"|"run", confidence: float, inference_ms: float}|
|  - Uses existing .venv with cached laya and vegaml weights                         |
+------------------------------------------------------------------------------------+
```

---

## 3. Backend API Specification (Python / FastAPI)

Located in `backend/server.py`. Runs on `http://localhost:8000`.

### 3.1 Endpoints

#### `GET /health`
Returns backend health status and model availability:
```json
{
  "status": "ok",
  "models": ["laya", "vega"]
}
```

#### `POST /predict`
Evaluates game state and returns an action decision:
* **Request Body**:
  ```json
  {
    "id": 12,
    "model": "laya",
    "state": {
      "distance_to_cactus": 18.5,
      "cactus_width": 3,
      "cactus_height": 2,
      "dino_is_grounded": true,
      "speed": 1.0,
      "latency_budget_ms": 45.0
    }
  }
  ```
* **Response Body**:
  ```json
  {
    "id": 12,
    "model": "laya",
    "action": "jump",
    "confidence": 0.89,
    "inference_ms": 26.4
  }
  ```

---

## 4. Frontend Specification (React + Vite)

Located in `frontend/`.

### 4.1 Game Engine & Physics
* **Canvas Resolution**: Logical coordinate space $800 \times 240$, dynamically scaled by `window.devicePixelRatio`.
* **Physics Timestep**: `requestAnimationFrame` with delta time clamping ($\Delta t \le 0.1\text{s}$) to avoid animation hitches.
* **Jump Dynamics**:
  - $v_0 = -13.5\text{ px/tick}$
  - $g = 0.72\text{ px/tick}^2$
  - Airtime $\approx 37$ frames at 60 FPS ($\approx 0.61$ seconds).
* **Hitbox & Clearance**:
  - Forgiving collision: Dino body hitbox centered at width 40, height 42.
  - Obstacles: Small cactus (width 24, height 38), Large cactus (width 30, height 50), Cluster (width 58, height 38).
  - Minimum gap: 400 pixels between obstacles.

### 4.2 UI & Telemetry Components
* **Terminal Retro Frame**: Glowing phosphor border, monospace labels (`font-mono`), scanline CRT overlay.
* **Canvas Viewport**: Crisp pixel Dino, running legs animation, ground scrolling with debris, jumping trajectory.
* **Telemetry Pane**:
  - Controller toggle: `[HUMAN]`, `[LAYA (0.8B)]`, `[VEGA (0.8B)]`.
  - Mode toggle: `[Round 1: Raw Async]`, `[Round 2: Fixed 45ms Hold]`.
  - Live statistics: current inference time, rolling p50, p90, p99, jitter ($\sigma$), and deadline misses ($>45\text{ms}$).
  - Real-time sparkline SVG chart of the last 30 decision latencies.

---

## 5. Directory Structure

```
layadino/
├── backend/
│   ├── server.py             # FastAPI REST server
│   └── requirements.txt      # fastapi, uvicorn, laya, vegaml
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx           # Main retro dashboard & hotkey listeners
│   │   ├── game/
│   │   │   ├── engine.ts     # Canvas 60 FPS Dino game engine & physics
│   │   │   ├── types.ts      # GameState, DinoObservation, Obstacle types
│   │   │   └── sprites.ts    # Pixel art render routines for Dino & Cacti
│   │   └── telemetry/
│   │       ├── api.ts        # Fetch client for POST /predict
│   │       ├── stats.ts      # Rolling percentiles (p50, p90, p99) and jitter
│   │       └── Sparkline.tsx # SVG latency timeline sparkline
│   └── ...
└── docs/
    └── superpowers/specs/2026-10-16-react-vite-dino-design.md
```
