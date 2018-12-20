all: build test

build:
	go get -v ./...
	go build ./...
	go install ./...

test:
	go test -coverprofile=coverage.out -timeout=60000ms -short -v ./...
