.PHONY: up build down clean logs dev init

up:
	docker compose up

build:
	docker compose up --build

down:
	docker compose down

clean:
	docker compose down -v --remove-orphans

logs:
	docker compose logs -f entropix

dev:
	docker compose up --build --watch entropix

init:
	# Install poetry
	curl -sSL https://install.python-poetry.org | python3 -
	# Install rust to build tiktoken
	curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
	# Poetry install
	poetry install
	# Download weights
	poetry run python download_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --out-dir weights/1B-Instruct