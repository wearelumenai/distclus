# Distclus

> Distributed online distance based clustering library

## Introduction

This library implements the concepts and theoretical results described in the article https://hal.inria.fr/hal-01264233.

## Installation

```bash
$ make build
```

## Test

```bash
$ make test
```

## Main abstractions

The distclus library intend to be polymorphic :
any element type can be used provided a distance and a barycenter can be computed between 2 elements.

A data type that can be clustered must implement the ```core.Elemt``` interface.
Distance and barycenter are calculated using a suitable object that implements ```core.Space```.

An algorithm is represented by a ```core.Algo``` object.
Actually several algorithm are provided by the library and more can be implemented by providing a specific
implementation to the ```core.Algo``` object.
An implementation is an objects that respects the ```core.Impl``` interface.

Three implementations are provided :
 - kmeans is built with the ```kmeans.NewAlgo``` constructor
 - mcmc is built with the ```mcmc.NewAlgo``` constructor
 - streaming is built with the ```streaming.NewAlgo``` constructor
 
Constructors need at least :
 - a configuration object which holds the algorithm parameters
 - a ```core.Space``` object used for distance and center computations
 
The result of a clustering is of type ```core.Clust``` which is an array of ```core.Elemt``` with dedicated methods.

## Configuration

Algorithm are configured with configuration objects. A minimal configuration for the MCMC algorithm requires 
two objects, one for the Metropolis Hastings and the other for the alteration distribution:

```go
package main
import "distclus/mcmc"

var conf = mcmc.Conf{
	InitK: 1,
	Amp:   100,
	B:     1,
}

var tConf = mcmc.MultivTConf{
	Dim:   2,
	Nu:    3,
}
```

where :
 - ```InitK``` is the starting number of clusters
 - ```Amp``` and ```B``` are used in the Metropolis Hastings accept ratio computation
 - ```Dim``` and ```Nu``` are used by the alteration distribution
 
For more information on setting these parameters refer to https://hal.inria.fr/hal-01264233

## Build the algorithm

The algorithm is built using the ```mcmc.NewAlgo``` function. It takes the following parameters :
 - ```conf mcmc.Conf``` : configuration object
 - ```space core.Space``` : distance and barycenter computation
 - ```data []core.Elemt``` : observations known at build time if any (```nil``` otherwise)
 - ```initializer kmeans.Initializer``` : a functor that returns the starting centers
 - ```distrib mcmc.Distrib``` : alteration distribution
 
 ```go
package main
import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
)

func Build(conf mcmc.Conf, tConf mcmc.MultivTConf, data []core.Elemt) (algo *core.Algo, space euclid.Space) {
	space = euclid.NewSpace(euclid.Conf{})
	var distrib = mcmc.NewMultivT(tConf) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, data, kmeans.PPInitializer, distrib)
	return
}
```

## Run and feed

The algorithm can be run in two modes :
 - synchronous : all data must be pushed before stating the algorithm
 - asynchronous (or online) : further data can be pushed after the algorithm is statrted
 
The following function starts the algorithm in asynchronous mode then pushes the observations

```go
package main
import (
	"time"
	"distclus/core"
)

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
	err = algo.Run(true) // run the algorithm in background
	for i := 0; i<len(observations) && err == nil; i++ {
		err = algo.Push(observations[i])
	}
	return
}
```

*Note :* when the algorithm starts, it first initializes the starting centers.
The number of centroids is given by the parameter ```InitK``` of the ```mcmc.Conf``` configuration object (see above).
Thus at least ```InitK``` observations must be given at construction time or pushed before the algorithm starts,
otherwise an error is returned by the ```Run``` method.

## Prediction

Once the algorithm was started, either in synchronous or online mode, 
the ```Predict``` method can be used to make predictions.
The following function uses predictions to calculate the root mean squared error 
for observations which real output is known.

```go
package main
import (
	"distclus/core"
	"distclus/euclid"
	"math"
)

func RMSE(algo *core.Algo, observations []core.Elemt, output []core.Elemt, space euclid.Space) float64 {
	var mse = 0.
	for i := range observations {
		var prediction, _, _ = algo.Predict(observations[i])
		var dist = space.Dist(prediction, output[i])
		mse += dist * dist / float64(len(observations))
	}
	return math.Sqrt(mse)
}
```

The following functions help in evaluating the algorithm.
The ```core.Clust``` object method ```MapLabel``` is used to compute the real output (see below: Advanced usage).

```go
package main
import (
	"distclus/core"
	"distclus/euclid"
)

func Eval(algo *core.Algo, centers core.Clust, observations []core.Elemt, space euclid.Space) (result core.Clust, rmse float64, err error) {
	var output = getOutput(centers, observations, space)
	rmse = RMSE(algo, observations, output, space)
	result, err = algo.Centroids()
	return
}

func getOutput(centers core.Clust, observations []core.Elemt, space euclid.Space) (output []core.Elemt) {
	var labels = centers.MapLabel(observations, space)
	output = make([]core.Elemt, len(labels))
	for i := range labels {
		output[i] = centers[labels[i]]
	}
	return
}
```

## Sample data

For testing purpose we need data. The following function returns real centers with a train and a test set.

```go
package main
import (
	"distclus/core"
	"golang.org/x/exp/rand"
)

func Sample() (core.Clust, []core.Elemt, []core.Elemt) {
	var centers = core.Clust(
		[]core.Elemt{
			[]float64{1.4, 0.7},
			[]float64{7.6, 7.6},
		})
	var observations = make([]core.Elemt, 1000)
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
	return centers, observations[:800], observations[800:]
}
```

## Putting all together

We will now glue the pieces we have built so far. The algorithm must be closed to stop online execution.
In the following example it is deferred to the end of the function.

```go
package main
import (
	"fmt"
	"time"
)

func Example() {
	var centers, train, test = Sample()
	
	var algo, space = Build(conf, tConf, train[:10])
	defer algo.Close()

	var errRun = RunAndFeed(algo, test[10:])

	if errRun == nil {
		time.Sleep(300 * time.Millisecond) // let the background algorithm converge
		var result, rmse, errEval = Eval(algo, centers, test, space)
		fmt.Printf("%v %v %v\n", len(result) < 4, rmse < 1, errEval)
	}
	// Output: true true <nil>
}
```

The timer is used to let the background algorithm converge. In a real life situation of course this is not needed:
The online algorithm will be closed only when the service is shutdown
and data will be pushed gradually when they arrive.

## Advanced usage

### ```core.Clust``` struct
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
var labels = centers.ParMapLabel(observations, space, norm)
```
This is useful when the clustering is done online because the centers are continually changing.

### ```OnlineClust``` interface

The ```OnlineClust``` interface is implemented by the ```core.Algo``` struct. We have covered almost all methods so far.
 - ```Centroids() (Clust, error)```
 - ```Push(elemt Elemt) error```
 - ```Predict(elemt Elemt) (Elemt, int, error)```
 - ```Run(async bool) error```
 - ```Close() error```
 - ```RuntimeFigures() (map[string]float64, error)```
 
Algorithms may return specific figures that describes their running state. These can be obtained by the
```RuntimeFigures``` method.
```go
var rt = algo.RuntimeFigures()
for name, value := range rt {
	fmt.Printf("%v: %v\n", name, value)
}
```

### ```LateDistrib``` struct

The ```mcmc.MutlivT``` implements a multivariate T distribution. The dimension of the data must be known and set
in the ```mcmc.MultivTConf``` configuration object. In some situation this information is not known until the first
data arrive. This can be handled using a ```LateDistrib``` instance. It implements the ```Distrib``` interface and
its responsibility is to initialize and wrap another ```Distrib``` instance. The ```Build``` function above
may be modified like this:
```go
package main
import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
)

func Build(conf mcmc.Conf, tConf mcmc.MultivTConf, data []core.Elemt) (algo *core.Algo, space euclid.Space) {
	space = euclid.NewSpace(euclid.Conf{})
	var buildDistrib = func(data core.Elemt) mcmc.Distrib {
		tConf.Dim = space.Dim([]core.Elemt{data})
		return mcmc.NewMultivT(tConf) 
	}
	var distrib = mcmc.NewLateDistrib(buildDistrib) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, data, kmeans.PPInitializer, distrib)
	return
}
```

The functor passed to ```mcmc.NewLateDistrib``` is executed only once with minimal locking to ensure parallel safety.
