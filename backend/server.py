import json
import os
import sys
import threading
import time
from typing import Any, Dict

import fastapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Suppress CUDA-specific fallback warnings on macOS / non-CUDA environments
logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)

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
        "instructions": "Chrome Dino runner. Choose whether to jump or run.",
        "criteria": {
            "jump": "Cactus is dangerously close in front of Dino. Leap immediately.",
            "run": "Track ahead is clear or cactus is far away. Keep running.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": "Chrome Dino runner. Choose whether to jump or run.",
        "criteria": {
            "jump": "Cactus is dangerously close in front of Dino. Leap immediately.",
            "run": "Track ahead is clear or cactus is far away. Keep running.",
        },
    }
}


def get_laya():
    global laya_model
    if laya_model is None:
        import laya
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        laya_model = laya.load("convaiinnovations/laya", device=device)
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import vegaml
        import torch
        # On Apple Silicon without CUDA, CPU runs the PyTorch chunked delta rule kernels
        # ~5.4x faster than MPS (12s vs 67s) due to vector SIMD cache locality.
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
        dist = float(state.get("distance_to_cactus", 999.0))
        grounded = bool(state.get("dino_is_grounded", True))
        budget = float(state.get("latency_budget_ms", 0.0))
        speed = float(state.get("speed", 1.0))

        # Critical danger close is within 7.0 cells when Dino is grounded
        status = "danger close" if (dist <= 7.0 and grounded and dist > 0) else "distant"

        if model_name == "vega":
            v = get_vega()
            formatted_state = f"Cactus distance: {dist:.1f} cells. Status: {status}. Grounded: {'yes' if grounded else 'no'}. Latency budget: {budget:.0f}ms."
            res = v.decide(formatted_state, VEGA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("confidence", 0.9))
        else:
            l = get_laya()
            formatted_state = {
                "cactus_distance": f"{dist:.1f} cells",
                "status": status,
                "grounded": "yes" if grounded else "no",
            }
            if budget > 0:
                formatted_state["latency_budget"] = f"{budget:.0f}ms"
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
