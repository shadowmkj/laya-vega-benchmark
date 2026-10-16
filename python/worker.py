#!/usr/bin/env python3
"""Asynchronous model worker for Chrome Dino game.

Reads line-delimited JSON requests from stdin, invokes either Laya or Vega decision
model, and outputs line-delimited JSON responses to stdout.
"""

import json
import os
import sys
import time
import warnings

# Suppress progress bars and warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

laya_model = None
vega_model = None

LAYA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino game. Pick JUMP, DUCK, or RUN to avoid crashing into obstacles.",
        "criteria": {
            "jump": "Ground obstacle (cactus or low bird) approaching ahead. Leap over it.",
            "duck": "Mid-height flying bird approaching at head level. Duck underneath it.",
            "run": "Runway is clear, obstacle is far away, or high bird will pass overhead safely.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Avoid obstacles in Chrome Dino by jumping, ducking, or running.",
        "criteria": {
            "jump": "Ground obstacle (cactus or low bird) is close and Dino must jump.",
            "duck": "Flying bird at head height is close and Dino must duck underneath.",
            "run": "Runway is clear, obstacle is far away, or high bird will clear overhead.",
        },
    }
}


def log_worker(msg: str):
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] [worker] {msg}\n")
    sys.stderr.flush()


def get_laya():
    global laya_model
    if laya_model is None:
        log_worker("[*] Loading Laya model ('convaiinnovations/laya')...")
        t0 = time.perf_counter()
        import laya
        laya_model = laya.load("convaiinnovations/laya")
        log_worker(f"[+] Laya model loaded successfully in {time.perf_counter() - t0:.2f}s")
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import torch
        # On Apple Silicon without CUDA, CPU runs chunked delta rule kernels much faster than MPS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_worker(f"[*] Loading Vega model on compute device: {device}...")
        t0 = time.perf_counter()
        import vegaml
        vega_model = vegaml.load(device=device)
        log_worker(f"[+] Vega model loaded successfully in {time.perf_counter() - t0:.2f}s on {device}")
    return vega_model


def decide_laya(state: dict) -> tuple[str, float]:
    model = get_laya()
    dist = float(state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0)))
    grounded = bool(state.get("dino_is_grounded", True))
    ducking = bool(state.get("dino_is_ducking", False))
    speed = float(state.get("speed", 1.0))
    width = state.get("cactus_width", 2)
    budget = float(state.get("latency_budget_ms", 0.0))
    obs_type = str(state.get("obstacle_type", "cactus")).lower()
    altitude = str(state.get("obstacle_altitude", "ground")).lower()

    if dist <= 7.5 and dist > 0:
        if obs_type == "pterodactyl" and altitude == "mid":
            status = "danger head-level bird: duck required"
        elif altitude == "high":
            status = "safe overhead bird: run"
        else:
            status = "danger ground obstacle: jump required"
    else:
        status = "distant"

    formatted_state = {
        "obstacle": f"{obs_type} ({altitude})",
        "distance": f"{dist:.1f} cells",
        "status": status,
        "grounded": "yes" if grounded else "no",
        "ducking": "yes" if ducking else "no",
        "speed": f"{speed:.2f}",
        "width": width,
    }
    if budget > 0:
        formatted_state["latency_budget"] = f"{budget:.0f}ms"

    res = model.predict(state=formatted_state, questions=LAYA_QUESTION)
    ans = res.get("answers", {}).get("action", {})
    action = ans.get("choice", "run")
    confidence = float(ans.get("answer_confidence", ans.get("confidence", 0.9)))
    return action, confidence


def decide_vega(state: dict) -> tuple[str, float]:
    model = get_vega()
    dist = float(state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0)))
    width = state.get("cactus_width", 2)
    grounded = bool(state.get("dino_is_grounded", True))
    ducking = bool(state.get("dino_is_ducking", False))
    speed = float(state.get("speed", 1.0))
    budget = float(state.get("latency_budget_ms", 0.0))
    obs_type = str(state.get("obstacle_type", "cactus")).lower()
    altitude = str(state.get("obstacle_altitude", "ground")).lower()

    if dist <= 7.5 and dist > 0:
        if obs_type == "pterodactyl" and altitude == "mid":
            status = "danger head-level bird: duck required"
        elif altitude == "high":
            status = "safe overhead bird: run"
        else:
            status = "danger ground obstacle: jump required"
    else:
        status = "distant"

    formatted_state = (
        f"Obstacle: {obs_type} at altitude {altitude}. Distance: {dist:.1f} cells. "
        f"Status: {status}. Width: {width}. Speed: {speed:.2f}. "
        f"Grounded: {grounded}. Ducking: {ducking}. Latency budget: {budget:.0f}ms."
    )
    res = model.decide(formatted_state, VEGA_QUESTION)
    ans = res.get("answers", {}).get("action", {})
    action = ans.get("choice", "run")
    confidence = float(ans.get("confidence", 0.9))
    return action, confidence


def main():
    # Signal readiness immediately so parent process starts with zero delay
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = 0
        try:
            req = json.loads(line)
            req_id = req.get("id", 0)
            model_name = req.get("model", "laya").lower()
            state = req.get("state", {})

            t0 = time.perf_counter()
            if model_name == "vega":
                action, conf = decide_vega(state)
            else:
                action, conf = decide_laya(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            resp = {
                "id": req_id,
                "model": model_name,
                "action": action.lower(),
                "confidence": conf,
                "inference_ms": round(elapsed_ms, 2),
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "id": req_id,
                "error": str(e),
                "action": "run",
                "confidence": 0.0,
                "inference_ms": 0.0,
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
