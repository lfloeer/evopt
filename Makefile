DOCKER_IMAGE := andig/evopt

default: build docker-build

build::
	go generate ./...

test::
	uv run pytest

install::
	uv sync

upgrade::
	uv lock --upgrade
	uv export --no-dev --no-hashes --no-emit-project > requirements.txt

lint::
	uv run ruff format
	uv run ruff check --fix

run::
	uv run python -m evopt.app

docker: docker-build docker-run

docker-build::
	docker buildx build . --tag $(DOCKER_IMAGE) --push

docker-run::
	docker run -p 7050:7050 -it $(DOCKER_IMAGE)

fly::
	fly deploy --local-only
