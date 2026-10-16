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
        "instructions": "Chrome Dino game survival action. Choose whether to jump, duck, or run.",
        "criteria": {
            "jump": "A cactus on the ground or low hazard is close ahead. Dino must jump over it. Ducking is fatal against cacti.",
            "duck": "A flying bird/pterodactyl at head-height is close ahead. Dino must duck underneath it. Never duck for a cactus.",
            "run": "The runway ahead is clear, obstacle is distant, or high bird flies safely overhead. Keep running.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino game survival action. Choose whether to jump, duck, or run.",
        "criteria": {
            "jump": "A cactus on the ground or low hazard is close ahead. Dino must jump over it. Ducking is fatal against cacti.",
            "duck": "A flying bird/pterodactyl at head-height is close ahead. Dino must duck underneath it. Never duck for a cactus.",
            "run": "The runway ahead is clear, obstacle is distant, or high bird flies safely overhead. Keep running.",
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


def decide_laya(state: dict) -> tuple[str, float, dict]:
    model = get_laya()
    dist = float(state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0)))
    grounded = bool(state.get("dino_is_grounded", True))
    ducking = bool(state.get("dino_is_ducking", False))
    speed = float(state.get("speed", 1.0))
    width = state.get("cactus_width", 2)
    budget = float(state.get("latency_budget_ms", 0.0))
    obs_type = str(state.get("obstacle_type", "cactus")).lower()
    altitude = str(state.get("obstacle_altitude", "ground")).lower()

    if obs_type == "cactus":
        obstacle_desc = "cactus on the ground"
    elif obs_type == "pterodactyl":
        if altitude == "mid":
            obstacle_desc = "flying pterodactyl at head level"
        elif altitude == "high":
            obstacle_desc = "flying pterodactyl high overhead"
        else:
            obstacle_desc = "flying pterodactyl low to the ground"
    else:
        obstacle_desc = f"{obs_type} ({altitude})"

    if 0 < dist <= 8.5:
        proximity = f"dangerously close ({dist:.1f} cells ahead)"
    else:
        proximity = f"distant ({dist:.1f} cells ahead)"

    formatted_state = {
        "obstacle": obstacle_desc,
        "distance": proximity,
        "dino_state": "running on the ground" if grounded else "airborne",
    }
    if budget > 0:
        formatted_state["latency_budget"] = f"{budget:.0f}ms"

    res = model.predict(state=formatted_state, questions=LAYA_QUESTION)
    ans = res.get("answers", {}).get("action", {})
    action = ans.get("choice", "run")
    confidence = float(ans.get("answer_confidence", ans.get("confidence", 0.9)))
    raw_probs = ans.get("probabilities", {})
    probabilities = {"jump": 0.0, "duck": 0.0, "run": 0.0}
    for k, v in raw_probs.items():
        probabilities[str(k).lower()] = round(float(v), 4)
    if sum(probabilities.values()) == 0.0:
        probabilities[action.lower()] = round(confidence, 4)
        rem = round((1.0 - confidence) / 2.0, 4) if confidence < 1.0 else 0.0
        for k in probabilities:
            if k != action.lower():
                probabilities[k] = max(0.0, rem)

    return action, confidence, probabilities


def decide_vega(state: dict) -> tuple[str, float, dict]:
    model = get_vega()
    dist = float(state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0)))
    width = state.get("cactus_width", 2)
    grounded = bool(state.get("dino_is_grounded", True))
    ducking = bool(state.get("dino_is_ducking", False))
    speed = float(state.get("speed", 1.0))
    budget = float(state.get("latency_budget_ms", 0.0))
    obs_type = str(state.get("obstacle_type", "cactus")).lower()
    altitude = str(state.get("obstacle_altitude", "ground")).lower()

    if obs_type == "cactus":
        obstacle_desc = "cactus on the ground"
    elif obs_type == "pterodactyl":
        if altitude == "mid":
            obstacle_desc = "flying pterodactyl at head level"
        elif altitude == "high":
            obstacle_desc = "flying pterodactyl high overhead"
        else:
            obstacle_desc = "flying pterodactyl low to the ground"
    else:
        obstacle_desc = f"{obs_type} ({altitude})"

    if 0 < dist <= 8.5:
        proximity = f"dangerously close ({dist:.1f} cells ahead)"
    else:
        proximity = f"distant ({dist:.1f} cells ahead)"

    formatted_state = (
        f"Obstacle: {obstacle_desc}. Distance: {proximity}. "
        f"Dino is {'grounded and running' if grounded else 'airborne'}. "
        f"Latency budget: {budget:.0f}ms."
    )
    res = model.decide(formatted_state, VEGA_QUESTION)
    ans = res.get("answers", {}).get("action", {})
    action = ans.get("choice", "run")
    confidence = float(ans.get("confidence", 0.9))
    raw_probs = ans.get("probs", {})
    probabilities = {"jump": 0.0, "duck": 0.0, "run": 0.0}
    for k, v in raw_probs.items():
        probabilities[str(k).lower()] = round(float(v), 4)
    if sum(probabilities.values()) == 0.0:
        probabilities[action.lower()] = round(confidence, 4)
        rem = round((1.0 - confidence) / 2.0, 4) if confidence < 1.0 else 0.0
        for k in probabilities:
            if k != action.lower():
                probabilities[k] = max(0.0, rem)

    return action, confidence, probabilities


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
                action, conf, probs = decide_vega(state)
            else:
                action, conf, probs = decide_laya(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            resp = {
                "id": req_id,
                "model": model_name,
                "action": action.lower(),
                "confidence": conf,
                "probabilities": probs,
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
                "probabilities": {"jump": 0.0, "duck": 0.0, "run": 1.0},
                "inference_ms": 0.0,
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
