FROM golang:alpine

WORKDIR /go/src/distclus

COPY . /go/src/distclus/pkg/

RUN make
