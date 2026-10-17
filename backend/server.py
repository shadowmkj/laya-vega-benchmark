import atexit
import json
import logging
import os
import queue
import sys
import threading
import time
from typing import Any

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
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(sh)

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


def log_model_memory_loaded(model_name: str, device: str, elapsed_s: float):
    import torch

    mem_lines = []
    if torch.cuda.is_available() and "cuda" in str(device).lower():
        alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        try:
            dev_name = torch.cuda.get_device_name(0)
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (
                1024 * 1024
            )
            pct = (alloc_mb / total_vram_mb * 100) if total_vram_mb > 0 else 0
            mem_lines.append(f"  • Compute Device:    {device} ({dev_name})")
            mem_lines.append(
                f"  • VRAM Allocated:    {alloc_mb:,.1f} MB / {total_vram_mb:,.1f} MB ({
                    pct:.1f}%)"
            )
            mem_lines.append(f"  • VRAM Reserved:     {res_mb:,.1f} MB")
        except Exception:
            mem_lines.append(f"  • Compute Device:    {device}")
            mem_lines.append(
                f"  • VRAM Allocated:    {alloc_mb:,.1f} MB (Reserved: {
                    res_mb:,.1f} MB)"
            )
    else:
        mem_lines.append(f"  • Compute Device:    {device}")

    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_lines.append(f"  • Process RAM (RSS): {rss_mb:,.1f} MB")
    except Exception:
        pass

    banner = [
        "",
        "================================================================================",
        f" [✓] MODEL LOADED TO MEMORY: {model_name.upper()} (0.8B)",
        *mem_lines,
        f"  • Load Duration:     {elapsed_s:.2f} seconds",
        f"  • Status:            ONLINE & READY FOR INFERENCE",
        "================================================================================",
        "",
    ]
    banner_text = "\n".join(banner)
    logger.info(banner_text)
    # Direct write to stdout ensures instant terminal display without stream buffering
    sys.stdout.write(banner_text + "\n")
    sys.stdout.flush()


@app.on_event("startup")
def on_startup():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "FastAPI Decision Backend initialized. PyTorch compute target: %s", device
    )
    logger.info(
        "Model status: Laya=%s, Vega=%s",
        "Loaded" if laya_model else "Standby",
        "Loaded" if vega_model else "Standby",
    )

    # Optional immediate pre-load to memory on startup via PRELOAD_MODEL env
    preload = os.getenv("PRELOAD_MODEL", "").strip().lower()
    if preload in ("1", "true", "laya"):
        logger.info(
            "[*] Pre-loading Laya model to memory on startup (PRELOAD_MODEL set)..."
        )
        get_laya()
    elif preload == "vega":
        logger.info(
            "[*] Pre-loading Vega model to memory on startup (PRELOAD_MODEL set)..."
        )
        get_vega()
    elif preload in ("all", "both"):
        logger.info(
            "[*] Pre-loading all models to memory on startup (PRELOAD_MODEL set)..."
        )
        get_laya()
        get_vega()


LAYA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": (
            "Select the correct maneuver for the dinosaur based on where the path is blocked:\n"
            "• Ground track blocked -> jump over it\n"
            "• Head-height airspace blocked -> duck / crouch underneath it\n"
            "• Airspace and ground clear -> run forward"
        ),
        "criteria": {
            "jump": "Ground track is blocked (cactus or low hazard). Leap upward into the open air above.",
            "duck": "Head-height airspace is blocked (mid-flight bird). Crouch low to duck underneath the obstacle.",
            "run": "Airspace at runner height is completely clear. Continue running upright forward.",
        },
    }
}

VEGA_QUESTION = {
    "action": {
        "type": "choice",
        "instructions": (
            "Select the correct maneuver for the dinosaur based on where the path is blocked:\n"
            "• Ground track blocked -> jump over it\n"
            "• Head-height airspace blocked -> duck / crouch underneath it\n"
            "• Airspace and ground clear -> run forward"
        ),
        "criteria": {
            "jump": "Ground track is blocked (cactus or low hazard). Leap upward into the open air above.",
            "duck": "Head-height airspace is blocked (mid-flight bird). Crouch low to duck underneath the obstacle.",
            "run": "Airspace at runner height is completely clear. Continue running upright forward.",
        },
    }
}


def get_laya():
    global laya_model
    if laya_model is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "[*] Loading Laya model ('convaiinnovations/laya') on compute device: %s...",
            device,
        )
        t_start = time.perf_counter()
        import laya

        laya_model = laya.load("convaiinnovations/laya", device=device)
        elapsed = time.perf_counter() - t_start
        log_model_memory_loaded("convaiinnovations/laya", device, elapsed)
    return laya_model


def get_vega():
    global vega_model
    if vega_model is None:
        import torch

        # NOTE: On Apple Silicon without CUDA, CPU runs the PyTorch
        # chunked delta rule kernels ~5.4x faster than MPS (12s vs 67s)
        # due to vector SIMD cache locality.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[*] Loading Vega model on compute device: %s...", device)
        t_start = time.perf_counter()
        import vegaml

        vega_model = vegaml.load(device=device)
        elapsed = time.perf_counter() - t_start
        log_model_memory_loaded("vegaml", device, elapsed)
    return vega_model


# Action cost weights: run=0, duck=1, jump=3 (jumping has highest physical commitment & mid-air vulnerability)
ACTION_COST_WEIGHTS = {"run": 0.0, "duck": 1.0, "jump": 3.0}
DEFAULT_COST_LAMBDA = float(os.getenv("COST_LAMBDA", "0.08"))


class AsyncJsonLogger:
    """Zero-overhead asynchronous JSON Lines logger.

    Offloads JSON serialization and disk writes completely from the critical model inference
    path to a background daemon worker thread consuming from a bounded in-memory queue.
    """

    def __init__(self, log_filepath: str, max_queue_size: int = 10000):
        self.log_filepath = log_filepath
        self.disabled = os.getenv("DISABLE_MODEL_LOG", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

        if not self.disabled:
            log_dir = os.path.dirname(os.path.abspath(self.log_filepath))
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.worker_thread = threading.Thread(
                target=self._worker, name="AsyncJsonLoggerWorker", daemon=True
            )
            self.worker_thread.start()
            atexit.register(self.shutdown)
            logger.info("Async model JSON logger active -> %s", self.log_filepath)

    def log(self, record: dict[str, Any]):
        """Enqueue record with zero blocking (< 5µs). Drops under extreme backpressure rather than stalling inference."""
        if self.disabled:
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass

    def _worker(self):
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                while not self.stop_event.is_set():
                    try:
                        record = self.queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    records = [record]
                    while len(records) < 50:
                        try:
                            records.append(self.queue.get_nowait())
                        except queue.Empty:
                            break

                    lines = []
                    for r in records:
                        try:
                            lines.append(json.dumps(r, default=str))
                        except Exception:
                            pass

                    if lines:
                        f.write("\n".join(lines) + "\n")
                        f.flush()

                    for _ in records:
                        self.queue.task_done()
        except Exception as e:
            logger.error("AsyncJsonLogger worker encountered an error: %s", e)

    def shutdown(self):
        self.stop_event.set()
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                while not self.queue.empty():
                    try:
                        record = self.queue.get_nowait()
                        f.write(json.dumps(record, default=str) + "\n")
                        self.queue.task_done()
                    except Exception:
                        break
                f.flush()
        except Exception:
            pass


MODEL_LOG_FILE = os.getenv("MODEL_LOG_FILE", "logs/model_io.jsonl")
async_logger = AsyncJsonLogger(MODEL_LOG_FILE)


class PredictRequest(BaseModel):
    id: int = 0
    model: str = "laya"
    state: dict[str, Any]
    # Scaling factor for action cost penalty (0.0 to disable)
    cost_lambda: float = DEFAULT_COST_LAMBDA


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
        dist = float(
            state.get("distance_to_cactus", state.get("distance_to_obstacle", 999.0))
        )
        grounded = bool(state.get("dino_is_grounded", True))
        ducking = bool(state.get("dino_is_ducking", False))
        budget = float(state.get("latency_budget_ms", 0.0))
        obs_type = str(state.get("obstacle_type", "cactus")).lower()
        altitude = str(state.get("obstacle_altitude", "ground")).lower()

        logger.debug(
            "Predict req #%s [%s]: obs=%s alt=%s dist=%.1f grounded=%s ducking=%s budget=%.0fms",
            req.id,
            model_name,
            obs_type,
            altitude,
            dist,
            grounded,
            ducking,
            budget,
        )

        status_text = (
            "crouching low"
            if ducking
            else ("upright stance" if grounded else "airborne")
        )

        if dist > 14.0 or dist <= 0:
            formatted_state = {
                "ground_track": "clear and unobstructed",
                "airspace": "open and clear at runner height",
                "collision_risk": "none (track is clear)",
                "safe_passage": "running upright passes safely forward",
                "proximity": "clear path",
                "distance": f"{dist:.1f} cells ahead",
                "runner_stance": status_text,
            }
        elif obs_type == "cactus":
            formatted_state = {
                "ground_track": "blocked by ground cactus",
                "airspace": "open and clear above",
                "collision_risk": "staying on ground will collide with cactus on track",
                "safe_passage": "leaping upward jump maneuver clears cactus",
                "proximity": "imminent hazard"
                if dist <= 7.8
                else "approaching in distance",
                "distance": f"{dist:.1f} cells ahead",
                "runner_stance": status_text,
            }
        elif obs_type == "pterodactyl" and altitude == "mid":
            formatted_state = {
                "ground_track": "clear beneath",
                "airspace": "blocked at head height by mid-flight pterodactyl",
                "collision_risk": "running upright will cause head-level collision",
                "safe_passage": "crouching low duck maneuver slips underneath",
                "proximity": "imminent hazard"
                if dist <= 7.8
                else "approaching in distance",
                "distance": f"{dist:.1f} cells ahead",
                "runner_stance": status_text,
            }
        elif obs_type == "pterodactyl" and altitude == "high":
            formatted_state = {
                "ground_track": "clear and unobstructed",
                "airspace": "clear at runner height (bird flies high in upper sky)",
                "collision_risk": "none at runner height (bird flies harmlessly overhead)",
                "safe_passage": "running upright passes safely underneath",
                "proximity": "overhead passing"
                if dist <= 7.8
                else "approaching in distance",
                "distance": f"{dist:.1f} cells ahead",
                "runner_stance": status_text,
            }
        else:  # low pterodactyl or unknown ground hazard
            formatted_state = {
                "ground_track": "blocked by low pterodactyl skimming the ground",
                "airspace": "open and clear above",
                "collision_risk": "staying on ground will collide with hazard on track",
                "safe_passage": "leaping upward jump maneuver clears hazard",
                "proximity": "imminent hazard"
                if dist <= 7.8
                else "approaching in distance",
                "distance": f"{dist:.1f} cells ahead",
                "runner_stance": status_text,
            }

        if budget > 0:
            formatted_state["latency_budget"] = f"{budget:.0f}ms"

        if model_name == "vega":
            v = get_vega()
            res = v.decide(formatted_state, VEGA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("confidence", 0.9))
            raw_probs = ans.get("probs", {})
        else:
            l = get_laya()
            res = l.predict(state=formatted_state, questions=LAYA_QUESTION)
            ans = res.get("answers", {}).get("action", {})
            action = ans.get("choice", "run")
            conf = float(ans.get("answer_confidence", ans.get("confidence", 0.9)))
            raw_probs = ans.get("probabilities", {})

        probabilities = {
            "jump": 0.0,
            "duck": 0.0,
            "run": 0.0,
        }
        for k, v in raw_probs.items():
            probabilities[str(k).lower()] = round(float(v), 4)

        if sum(probabilities.values()) == 0.0:
            probabilities[action.lower()] = round(conf, 4)
            rem = round((1.0 - conf) / 2.0, 4) if conf < 1.0 else 0.0
            for k in probabilities:
                if k != action.lower():
                    probabilities[k] = max(0.0, rem)

        raw_model_action = action
        raw_model_conf = conf

        # Decision-Theoretic Cost-Weighted Utility Selection:
        # Score(a) = P(a) - cost_lambda * Cost(a)
        # For airborne hazards (mid/high pterodactyl): jumping is fatal, so jump=3, duck=1, run=0.
        # For ground hazards (cacti): jumping is mandatory (jump=0, run=2, duck=2).
        cost_lambda = (
            float(req.cost_lambda)
            if req.cost_lambda is not None
            else DEFAULT_COST_LAMBDA
        )
        is_airborne_hazard = obs_type == "pterodactyl" and altitude in ("mid", "high")

        utility_scores = {}
        for act_name, prob in probabilities.items():
            if is_airborne_hazard:
                cost = ACTION_COST_WEIGHTS.get(act_name, 0.0)
            elif obs_type == "cactus" or altitude == "ground":
                # Ground obstacles: jumping has 0 penalty; failing to jump carries penalty
                cost = 0.0 if act_name == "jump" else 2.0
            else:
                cost = 0.0
            utility_scores[act_name] = round(prob - (cost_lambda * cost), 4)

        if cost_lambda > 0:
            optimal_action = max(utility_scores.keys(), key=lambda a: utility_scores[a])
            action = optimal_action
            conf = probabilities[action]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Non-blocking async JSON logging (sub-millisecond handoff to background worker)
    async_logger.log(
        {
            "timestamp": round(time.time(), 3),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int((time.time() % 1) * 1000):03d}Z",
            "request_id": req.id,
            "model": model_name,
            "input": {
                "raw_state": req.state,
                "formatted_state": formatted_state,
                "question": VEGA_QUESTION if model_name == "vega" else LAYA_QUESTION,
                "cost_lambda": cost_lambda,
            },
            "output": {
                "raw_model_action": raw_model_action.lower(),
                "raw_model_confidence": round(raw_model_conf, 4),
                "probabilities": probabilities,
                "utility_scores": utility_scores,
                "final_action": action.lower(),
                "final_confidence": round(conf, 4),
                "inference_ms": round(elapsed_ms, 2),
            },
        }
    )

    logger.info(
        "Decision #%s [%s]: action=%s (%.1f%%) in %.1fms [obs=%s @ %.1f cells] options=%s utilities=%s",
        req.id,
        model_name.upper(),
        action.upper(),
        conf * 100.0,
        elapsed_ms,
        obs_type,
        dist,
        probabilities,
        utility_scores,
    )

    return {
        "id": req.id,
        "model": model_name,
        "action": action.lower(),
        "confidence": round(conf, 4),
        "probabilities": probabilities,
        "utility_scores": utility_scores,
        "inference_ms": round(elapsed_ms, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
