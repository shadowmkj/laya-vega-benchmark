import json
import os
import subprocess
import sys


def test_worker_pipe():
    python_bin = ".venv/bin/python" if os.path.exists(".venv/bin/python") else sys.executable
    print(f"Testing worker with: {python_bin}")

    proc = subprocess.Popen(
        [python_bin, "python/worker.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    # 1. Immediate handshake check
    ready_line = proc.stdout.readline()
    assert "ready" in ready_line, f"Unexpected handshake: {ready_line}"
    print(f"Handshake received: {ready_line.strip()}")

    # 2. Test Laya
    print("Testing Laya inference...")
    req_laya = {
        "id": 1,
        "model": "laya",
        "state": {
            "distance_to_cactus": 10.0,
            "cactus_width": 2,
            "cactus_height": 2,
            "dino_is_grounded": True,
            "speed": 1.0,
            "latency_budget_ms": 0.0,
        },
    }
    proc.stdin.write(json.dumps(req_laya) + "\n")
    proc.stdin.flush()
    resp_laya = json.loads(proc.stdout.readline())
    print(f"Laya response: {resp_laya}")
    assert resp_laya["id"] == 1
    assert resp_laya["action"] in ["jump", "run"]

    # 3. Test Vega
    print("Testing Vega inference...")
    req_vega = {
        "id": 2,
        "model": "vega",
        "state": {
            "distance_to_cactus": 8.0,
            "cactus_width": 2,
            "cactus_height": 2,
            "dino_is_grounded": True,
            "speed": 1.0,
            "latency_budget_ms": 45.0,
        },
    }
    proc.stdin.write(json.dumps(req_vega) + "\n")
    proc.stdin.flush()
    resp_vega = json.loads(proc.stdout.readline())
    print(f"Vega response: {resp_vega}")
    assert resp_vega["id"] == 2
    assert resp_vega["action"] in ["jump", "run"]

    # 4. Second Vega inference (warm)
    proc.stdin.write(json.dumps(req_vega) + "\n")
    proc.stdin.flush()
    resp_vega_warm = json.loads(proc.stdout.readline())
    print(f"Vega warm response: {resp_vega_warm}")
    print(f"Vega warm latency: {resp_vega_warm['inference_ms']}ms")

    proc.terminate()
    print("Worker test for BOTH Laya and Vega PASSED successfully!")


if __name__ == "__main__":
    test_worker_pipe()
