FROM golang:alpine

WORKDIR /go/src/distclus

COPY . /go/src/distclus/

RUN make
