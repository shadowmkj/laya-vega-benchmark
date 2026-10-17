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

    # 2. Predict Laya - Cactus Close (Must Jump)
    req_cactus = {
        "id": 1,
        "model": "laya",
        "state": {
            "distance_to_cactus": 5.0,
            "cactus_width": 2,
            "obstacle_type": "cactus",
            "obstacle_altitude": "ground",
            "dino_is_grounded": True,
            "speed": 1.0,
        },
    }
    res_cactus = client.post("/predict", json=req_cactus)
    assert res_cactus.status_code == 200
    data_cactus = res_cactus.json()
    assert data_cactus["action"] == "jump", f"Expected jump, got {data_cactus['action']}"
    print(f"Cactus test passed: {data_cactus}")

    # 3. Predict Laya - Pterodactyl Ground (Must Jump)
    req_ptero_ground = {
        "id": 2,
        "model": "laya",
        "state": {
            "distance_to_cactus": 5.0,
            "obstacle_type": "pterodactyl",
            "obstacle_altitude": "ground",
            "dino_is_grounded": True,
            "speed": 1.0,
        },
    }
    res_ptero_ground = client.post("/predict", json=req_ptero_ground)
    assert res_ptero_ground.status_code == 200
    data_ptero_ground = res_ptero_ground.json()
    assert data_ptero_ground["action"] == "jump", f"Expected jump, got {data_ptero_ground['action']}"
    print(f"Pterodactyl ground test passed: {data_ptero_ground}")

    # 4. Predict Laya - Pterodactyl Mid (Must Duck)
    req_ptero_mid = {
        "id": 3,
        "model": "laya",
        "state": {
            "distance_to_cactus": 5.0,
            "obstacle_type": "pterodactyl",
            "obstacle_altitude": "mid",
            "dino_is_grounded": True,
            "speed": 1.0,
        },
    }
    res_ptero_mid = client.post("/predict", json=req_ptero_mid)
    assert res_ptero_mid.status_code == 200
    data_ptero_mid = res_ptero_mid.json()
    assert data_ptero_mid["action"] == "duck", f"Expected duck, got {data_ptero_mid['action']}"
    print(f"Pterodactyl mid test passed: {data_ptero_mid}")

    # 5. Predict Laya - Pterodactyl High (Must Run)
    req_ptero_high = {
        "id": 4,
        "model": "laya",
        "state": {
            "distance_to_cactus": 5.0,
            "obstacle_type": "pterodactyl",
            "obstacle_altitude": "high",
            "dino_is_grounded": True,
            "speed": 1.0,
        },
    }
    res_ptero_high = client.post("/predict", json=req_ptero_high)
    assert res_ptero_high.status_code == 200
    data_ptero_high = res_ptero_high.json()
    assert data_ptero_high["action"] == "run", f"Expected run, got {data_ptero_high['action']}"
    print(f"Pterodactyl high test passed: {data_ptero_high}")

    # 6. Predict Laya - Distant Obstacle (Must Run)
    req_distant = {
        "id": 5,
        "model": "laya",
        "state": {
            "distance_to_cactus": 15.0,
            "obstacle_type": "cactus",
            "obstacle_altitude": "ground",
            "dino_is_grounded": True,
            "speed": 1.0,
        },
    }
    res_distant = client.post("/predict", json=req_distant)
    assert res_distant.status_code == 200
    data_distant = res_distant.json()
    assert data_distant["action"] == "run", f"Expected run, got {data_distant['action']}"
    print(f"Distant obstacle test passed: {data_distant}")

if __name__ == "__main__":
    test_api()
