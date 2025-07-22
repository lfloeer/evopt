default:: 

build::
	go generate ./...

server::
	python3 -m venv .venv && source .venv/bin/activate && python3 main.py

docker::
	docker buildx build . --tag andig/milp --push
