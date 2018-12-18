all: build test

build:
	go get -v ./...
	go build ./...
	go install ./...

test:
	go test -v ./...
