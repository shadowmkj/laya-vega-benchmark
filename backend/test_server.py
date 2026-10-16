import json
import sys
from fastapi.testclient import TestClient
from backend.server import app

def test_api():
    client = TestClient(app)
    # 1. Health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print(f"Health response: {res.json()}")

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
