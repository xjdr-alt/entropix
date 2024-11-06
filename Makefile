.PHONY: up build down clean logs

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