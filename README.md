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

### Main abstractions
The distclus library intend to be polymorphic :
any element type can be used provided a distance and a barycenter can be computed between 2 elements.

A data type than can be clustered must implement the ```core.Elemt``` interface. Distance and barycenter are calculated
using a corresponding object that implements ```core.Space```.

An algorithm is implemented by a ```core.Algo``` object which has the ```core.OnlineClust``` interface.
Actually several algorithm are provided by the library and more can be implemented by providing a specific
implementation to the ```core.Algo``` object. An implementation must follow the ```core.Impl``` interface.

Three implementations are provided :
 - kmeans using the ```kmeans.NewAlgo``` constructor
 - mcmc using the ```mcmc.NewAlgo``` constructor
 - streaming using the ```streaming.NewAlgo``` constructor
 
Constructors need at least :
 - a configuration object of the right type with the proper algorithm parameters
 - a ```core.Space``` object used for distance and center computations
 
The result of a clustering is of type ```core.Clust``` which is an array of ```core.Elemt``` with dedicated methods.

### Configuration
An algorithm is configured with a configuration object. A minimal configuration for the MCMC algorithm is
```go
package main

import "distclus/mcmc"

var conf = mcmc.Conf{
	InitK: 1,
	Dim:   2,
	Amp:   100,
	B:     1,
	Nu:    3,
}
```
where :
 - ```InitK``` is the starting number of clusters
 - ```Amp``` and ```B``` are used in the accept ratio computation
 - ```Dim``` and ```Nu``` are used for the alteration distribution
For more information about the parameters, please refer to https://hal.inria.fr/hal-01264233

### Run the algorithm

### A toy example

For testing purpose the following function can be used to generate trivial sample data :
```go
package main

import (
	"distclus/core"
	"golang.org/x/exp/rand"
)

func Sample() (centers core.Clust, observations []core.Elemt) {
	centers = core.Clust(
		[]core.Elemt{
			[]float64{1.4, 1.2},
			[]float64{3.6, 3.6},
		})
	observations = make([]core.Elemt, 1000)
	for i := range observations {
		var obs = make([]float64, 2)
		if rand.Intn(2) == 1 {
			copy(obs, centers[0].([]float64))
		} else {
			copy(obs, centers[1].([]float64))
		}
		for j := range obs {
			obs[j] += rand.Float64() - 1
		}
		observations[i] = obs
	}
	return
}
```

Create an online clustering algorithm `oc`:

```go
package main
import (
	"distclus/core"
	"distclus/factory"
)

conf := core.Conf{...}

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

## Advanced usage

Given clustering centers, it is possible to obtain internal loss and cardinality for each cluster :

```go
var space = euclid.NewSpace(euclid.Conf{})
var norm = 2.0
var losses, cards = centers.ReduceLoss(observations, space, norm)
```

The same could be performed in parallel :
```go
var space = euclid.NewSpace(euclid.Conf{})
var norm = 2.0
var degree = runtime.NumCPU()
var losses, cards = centers.ParReduceLoss(observations, space, norm)
```

In some cases you'll want to predict labels upon fixed cluster centers :
```go
var space = euclid.NewSpace(euclid.Conf{})
var labels = centers.MapLabel(observations, space, norm)
```
This is useful when the clustering is done online because the centers are continually changing.

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
