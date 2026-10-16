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
    assert data["action"] in ["jump", "duck", "run"]
    assert data["inference_ms"] > 0
    print(f"Backend API test passed! Response: {data}")

    # 3. Predict Laya with Mid Pterodactyl
    req_duck = {
        "id": 2,
        "model": "laya",
        "state": {
            "distance_to_cactus": 5.0,
            "obstacle_type": "pterodactyl",
            "obstacle_altitude": "mid",
            "dino_is_grounded": True,
            "dino_is_ducking": False,
            "speed": 1.0,
            "latency_budget_ms": 0.0,
        },
    }
    res_duck = client.post("/predict", json=req_duck)
    assert res_duck.status_code == 200
    data_duck = res_duck.json()
    assert data_duck["id"] == 2
    assert data_duck["action"] in ["jump", "duck", "run"]
    print(f"Backend API pterodactyl duck test passed! Response: {data_duck}")

if __name__ == "__main__":
    test_api()
