# Lag-Compensated Terminal Chrome Dino (Rust + Ratatui + Laya/Vega)
**Design Specification**

*Date: 2026-10-16*
*Status: Approved*

---

## 1. Overview & Objective

In real-time arcade games (like Chrome Dino or Flappy Bird), neural decision models face an **inference lag** bottleneck: when a model takes 20–40 ms to process an observation and return a jump command, the obstacle has already moved forward during the forward pass. Variance and jitter in inference latency cause mistimed jumps and unexpected deaths.

This project implements a terminal Chrome Dino game written in **Rust** using **`ratatui`** and **`crossterm`**, with model inference served asynchronously by **Python** running **Laya** and **Vega** typed decision models.

### Key Goals
1. **Playable Human Game**: A responsive, 30 FPS terminal Chrome Dino game playable via Spacebar / Arrow keys.
2. **Unified Agent Driver**: Switch seamlessly between `Human`, `Laya`, and `Vega` via hotkeys.
3. **Experiment Round 1 (Raw Async)**: Measure how variable inference latency and jitter cause missed jumps when actions are executed immediately upon receipt.
4. **Experiment Round 2 (Fixed 45ms Horizon Hold / Lag Compensation)**:
   - Provide the model with a projected obstacle state at $+45\text{ ms}$ (`latency_budget: 45ms`).
   - Hold decision execution until exactly $45\text{ ms}$ post-observation to eliminate timing jitter.
   - Benchmark latency variance (p50 vs p90 vs p99 jitter) between Laya (Transformer) and Vega (12-step symplectic integrator) under load.

---

## 2. System Architecture

```
+--------------------------------------------------------------------------------+
|                           RUST PROCESS (layadino)                              |
|                                                                                |
|  +--------------------------------------------------------------------------+  |
|  |                       Game Loop (Strict 30 FPS)                          |  |
|  |  - Physics update (Delta t = 33.3 ms)                                    |  |
|  |  - Crossterm event poll (Human input & hotkeys)                          |  |
|  |  - Ratatui terminal rendering                                            |  |
|  |  - Scheduled action queue (holds delayed actions for 45ms mode)          |  |
|  +--------------------------------------------------------------------------+  |
|         |                                                             ^        |
|     State Snapshot                                             Action Response |
|         v                                                             |        |
|  +---------------------+                                     +--------------+  |
|  | IPC Request Sender  |                                     | IPC Receiver |  |
|  +---------------------+                                     +--------------+  |
+------------|----------------------------------------------------------^--------+
             | stdin (JSON lines)                                       | stdout (JSON lines)
             v                                                          |
+-----------------------------------------------------------------------|--------+
|                      PYTHON WORKER (worker.py)                                 |
|                                                                                |
|  - Reads JSON line requests from stdin                                         |
|  - Dispatches to active model:                                                 |
|      * Laya (convaiinnovations/laya via laya.load)                             |
|      * Vega (vegaml.load)                                                      |
|  - Measures precise high-resolution inference duration                         |
|  - Emits JSON response to stdout                                               |
+--------------------------------------------------------------------------------+
```

---

## 3. Core Game Engine (Rust)

### 3.1 Timing & Tick Rate
* **Frame rate**: Strict 30 FPS ($33.\overline{3}\text{ ms}$ per tick).
* **Clock**: Uses `std::time::Instant`. Each frame calculates elapsed time; remaining frame budget is yielded/slept using a high-precision spin/sleep hybrid to prevent frame drift.
* **Decoupled execution**: Game loop **never blocks** on IPC or model responses.

### 3.2 Physics & Mechanics
* **Dino Coordinates**:
  - $X = 6$ (fixed horizontal anchor).
  - $Y \in [0.0, Y_{\max}]$ where $0.0$ is ground level.
* **Jump Dynamics**:
  - Initial upward velocity $v_0 = 4.2\text{ cells/frame}$.
  - Gravity $g = -1.2\text{ cells/frame}^2$.
  - Apex reached at frame 3–4; airtime lasts $\approx 14$ frames.
* **Obstacles (Cacti)**:
  - Moving leftwards at speed $v$ (starts at $1.0\text{ cell/frame} = 30\text{ cells/sec}$, slowly scaling with score).
  - Variations:
    - Small cactus: width 2, height 2.
    - Large cactus: width 3, height 3.
    - Cluster: width 4–5, height 2–3.
  - Spawning uses seedable PRNG (`fastrand` / `rand_chacha`) so test runs can be repeated deterministically.
* **Collision Detection**:
  - AABB intersection between Dino bounding box and cactus bounding boxes with a 1-character horizontal grace tolerance for fair gameplay.

---

## 4. IPC & Communication Protocol

The Rust binary launches `python -u worker.py` as a child process with piped standard I/O streams. Communication uses line-delimited JSON.

### 4.1 Request Format (Rust $\to$ Python)
```json
{
  "id": 42,
  "model": "laya",
  "mode": "fixed_horizon",
  "state": {
    "distance_to_cactus": 22.4,
    "cactus_width": 2,
    "cactus_height": 2,
    "dino_is_grounded": true,
    "speed": 1.0,
    "latency_budget_ms": 45.0
  }
}
```

### 4.2 Response Format (Python $\to$ Rust)
```json
{
  "id": 42,
  "action": "jump",
  "confidence": 0.942,
  "inference_ms": 26.85,
  "raw_readout": {
    "jump_score": 0.942,
    "run_score": 0.058
  }
}
```

---

## 5. Model Inference & Decision Formulations (Python)

### 5.1 Laya Integration
* **Model**: `laya.load("convaiinnovations/laya")`.
* **State format**: Dict with current Dino dynamics and upcoming cactus distance.
* **Question Schema**:
  ```python
  {
      "action": {
          "type": "choice",
          "instructions": "Chrome Dino game. Decide whether to JUMP or RUN.",
          "criteria": {
              "jump": "Obstacle is within jump window and Dino is grounded. Jump now to clear.",
              "run": "Obstacle is far or already cleared. Stay grounded."
          }
      }
  }
  ```

### 5.2 Vega Integration
* **Model**: `vegaml.load(device="mps")` (fallback to `cpu`).
* **Physics-based Decision Readout**: Evaluates energy wells for `jump` vs `run`.
* **Question Schema**:
  ```python
  {
      "action": {
          "type": "choice",
          "instructions": "Chrome Dino obstacle avoidance. Pick JUMP or RUN.",
          "criteria": {
              "jump": "Cactus is approaching and Dino must leap to avoid collision.",
              "run": "Ground is clear or cactus is beyond jump trigger distance."
          }
      }
  }
  ```

---

## 6. Experimental Modes & Lag Compensation

### Round 1: Raw Async (Uncompensated)
1. When an obstacle enters the trigger zone ($D \le 30$ cells), Rust captures the snapshot and sends it to the worker.
2. The game loop keeps advancing at 30 FPS.
3. The moment Python returns an action, Rust applies it immediately.
4. **Result**: If inference takes 32 ms, the cactus moved $\approx 1$ frame closer during calculation. Jitter causes the jump to trigger at inconsistent positions, inducing collisions.

### Round 2: Fixed 45ms Horizon Hold (Lag-Compensated)
1. Rust computes projected distance at $t + 45\text{ ms}$:
   $$d_{\text{projected}} = d_{\text{current}} - (v \times 0.045)$$
2. Rust sends $d_{\text{projected}}$ and `latency_budget_ms: 45.0` to the model.
3. When Python finishes (e.g. at 26 ms), Rust schedules the action in an internal event queue to execute at exactly $(t_{\text{request}} + 45\text{ ms})$.
4. If inference ever exceeds 45 ms (a deadline miss), a telemetry warning is recorded.
5. **Comparison Benchmark**: Measures p50, p90, p99, and $\sigma$ of latency between Vega's 12-step symplectic integration and Laya's Transformer pass.

---

## 7. Terminal User Interface (Ratatui)

### 7.1 Visual Layout
```
+------------------------------------------------------------------------------------+
|  SCORE: 00482   HI: 01240   SPEED: 1.15   FPS: 30.0   CONTROLLER: [VEGA (0.8B)]    |
+------------------------------------------------------------------------------------+
|                                                                                    |
|                                                                                    |
|                    (airborne)                                                      |
|                       __                                                           |
|                      / _)                                                          |
|             .-^^^-/ /                                                              |
|          __/       /                                    _|_         ||             |
|         <__.|_|-|_|                                    | | |       |  |            |
|  ___.._________________________________________________|_|_|_______|__|__________  |
|                                                                                    |
+--------------------------------------------------+---------------------------------+
| AGENT TELEMETRY & CONTROLS                       | INFERENCE LATENCY & JITTER      |
|                                                  |                                 |
| Mode: [Round 2: Fixed 45ms Hold]                 | Current: 24.3 ms                |
| In-Flight Action: JUMP (exec in 18ms)            | p50: 23.8 ms   p90: 28.1 ms     |
| Confidence: 95.8%                                | p99: 31.4 ms   Sigma: 2.1 ms    |
| Projected Obstacle Dist: 14.2 cells              | Deadline Misses (<45ms): 0      |
|                                                  | [|||||||||||||||||||||........] |
| Hotkeys: [Space] Jump  [m] Swap Model            |                                 |
|          [h] Toggle Round  [r] Reset  [q] Quit   |                                 |
+--------------------------------------------------+---------------------------------+
```

---

## 8. Development & Implementation Plan

1. **Step 1: Rust Game Scaffold & Engine**
   - Setup `Cargo.toml` with `ratatui`, `crossterm`, `serde`, `serde_json`.
   - Implement `Dino`, `Obstacle`, `World`, and jump physics.
   - Implement human playability via Space / Up arrow.
2. **Step 2: Python Inference Worker**
   - Create `python/worker.py` with standard I/O JSON protocol.
   - Integrate `laya` and `vegaml` loaders and decision endpoints.
   - Test standalone pipe requests.
3. **Step 3: Rust Async Driver & IPC**
   - Implement `IpcClient` with background thread and non-blocking channels.
   - Implement `LayaAgent` and `VegaAgent`.
4. **Step 4: Round 1 & Round 2 Lag Compensation & Telemetry**
   - Implement Raw execution vs Scheduled 45ms Horizon Hold.
   - Collect and render real-time p50/p90/p99 latency sparklines and jitter metrics.
5. **Step 5: Verification & End-to-End Testing**
   - Test human play, Laya auto-play, Vega auto-play, and verify latency stats under both modes.
