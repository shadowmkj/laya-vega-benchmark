run:
	uv run uvicorn backend.server:app --host=0.0.0.0 --port=8000
frontend:
	cd frontend && pnpm dev
sync:
	git stash && git pull mac && git stash pop
