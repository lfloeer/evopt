default:: 

build::
	go generate ./...

docker::
	docker buildx build . --tag andig/milp --push
