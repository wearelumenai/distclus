# Distclus

> Distributed online clustering library

## Installation

```bash
$ make build
```

## Test

```bash
$ make test
```

## How to

Create an online clustering algorithm `oc`:

```go
import ("distclus/factory")

conf := Conf{...}

space := factory.CreateSpace(space, conf)

oc := factory.CreateOC(name, conf, space, data, initializer, args...)

oc.Fit()
```

Where

- conf: algorithm configuration
- space: space name among "vectors", "series", etc.
- name: an algorithm name (among kmeans, mcmc, etc.)
- conf: configuration algorithm
- space: space interface dedicated to process distance between data and centroids
- initializer: algorithm initialization (random, given, etc.)
- args...: aditional parameters specific to algorithm (mcmc distrib, etc.)

## Contribute

### OnlineClust

According to a configuration and a space interface, the OnlineClust interface executes an algorithm.

#### Development

1. create a folder named with your algorithm name. I.e, `mcmc/` for the MCMC algorithm.
2. implement the interface `core.OnlineClust`. I.e., mcmc.Algo
3. add this implementation in the `factory.CreateOC` function in `factory/factory.go`

### Space

1. create a folder named with your space name. I.e, `vectors/` for the vectors space.
2. implement the interface `core.Space`. I.e., `vectors.Space`
3. add this implementation in the `factory.CreateSpace` function in `factory/factory.go`
