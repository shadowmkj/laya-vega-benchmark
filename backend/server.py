import json
import os
import sys
import threading
import time
from typing import Any, Dict
import logging

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Suppress CUDA-specific fallback warnings on macOS / non-CUDA environments
logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend")

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


@app.on_event("startup")
def on_startup():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("FastAPI Decision Backend initialized. PyTorch compute target: %s", device)
    logger.info("Model status: Laya=%s, Vega=%s (lazy-loaded on first request)", 
                "Loaded" if laya_model else "Standby",
                "Loaded" if vega_model else "Standby")


LAYA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino runner. Choose whether to jump, duck, or run.",
        "criteria": {
            "jump": "Ground obstacle (cactus or low bird) approaching ahead. Leap over it.",
            "duck": "Mid-height flying bird approaching at head level. Duck underneath it.",
            "run": "Track ahead is clear, obstacle is far away, or high bird will pass overhead safely.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino runner. Choose whether to jump, duck, or run.",
        "criteria": {
            "jump": "Ground obstacle (cactus or low bird) approaching ahead. Leap over it.",
            "duck": "Mid-height flying bird approaching at head level. Duck underneath it.",
            "run": "Track ahead is clear, obstacle is far away, or high bird will pass overhead safely.",
        },
    }
}


def get_laya():
    global laya_model
    if laya_model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[*] Loading Laya model ('convaiinnovations/laya') on compute device: %s...", device)
        t_start = time.perf_counter()
        import laya
        laya_model = laya.load("convaiinnovations/laya", device=device)
        logger.info("[+] Laya model loaded successfully in %.2fs on %s", time.perf_counter() - t_start, device)
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import torch
        # On Apple Silicon without CUDA, CPU runs the PyTorch chunked delta rule kernels
        # ~5.4x faster than MPS (12s vs 67s) due to vector SIMD cache locality.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[*] Loading Vega model on compute device: %s...", device)
        t_start = time.perf_counter()
        import vegaml
        vega_model = vegaml.load(device=device)
        logger.info("[+] Vega model loaded successfully in %.2fs on %s", time.perf_counter() - t_start, device)
    return vega_model


class PredictRequest(BaseModel):
    id: int = 0
    model: str = "laya"
    state: Dict[str, Any]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["laya", "vega"],
        "loaded": {
            "laya": laya_model is not None,
            "vega": vega_model is not None,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    model_name = req.model.lower()
    state = req.state

    with model_lock:
        dist = float(state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0)))
        grounded = bool(state.get("dino_is_grounded", True))
        ducking = bool(state.get("dino_is_ducking", False))
        budget = float(state.get("latency_budget_ms", 0.0))
        speed = float(state.get("speed", 1.0))
        obs_type = str(state.get("obstacle_type", "cactus")).lower()
        altitude = str(state.get("obstacle_altitude", "ground")).lower()

        # Danger status detection
        if dist <= 7.5 and dist > 0:
            if obs_type == "pterodactyl" and altitude == "mid":
                status = "danger head-level bird: duck required"
            elif altitude == "high":
                status = "safe overhead bird: run"
            else:
                status = "danger ground obstacle: jump required"
        else:
            status = "distant"

        logger.debug(
            "Predict req #%s [%s]: obs=%s alt=%s dist=%.1f grounded=%s budget=%.0fms",
            req.id, model_name, obs_type, altitude, dist, grounded, budget
        )

        if model_name == "vega":
            v = get_vega()
            formatted_state = (
                f"Obstacle: {obs_type} at altitude {altitude}. Distance: {dist:.1f} cells. "
                f"Status: {status}. Grounded: {'yes' if grounded else 'no'}. "
                f"Ducking: {'yes' if ducking else 'no'}. Latency budget: {budget:.0f}ms."
            )
            res = v.decide(formatted_state, VEGA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("confidence", 0.9))
        else:
            l = get_laya()
            formatted_state = {
                "obstacle": f"{obs_type} ({altitude})",
                "distance": f"{dist:.1f} cells",
                "status": status,
                "grounded": "yes" if grounded else "no",
                "ducking": "yes" if ducking else "no",
            }
            if budget > 0:
                formatted_state["latency_budget"] = f"{budget:.0f}ms"
            res = l.predict(state=formatted_state, questions=LAYA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("answer_confidence", ans.get("confidence", 0.9)))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Decision #%s [%s]: action=%s (%.1f%%) in %.1fms [obs=%s @ %.1f cells]",
        req.id, model_name.upper(), action.upper(), conf * 100.0, elapsed_ms, obs_type, dist
    )

    return {
        "id": req.id,
        "model": model_name,
        "action": action.lower(),
        "confidence": round(conf, 4),
        "inference_ms": round(elapsed_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
