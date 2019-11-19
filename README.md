# Distclus

> Multi-threaded online distance based clustering library

## Introduction

This library implements the concepts and theoretical results described in the article https://hal.inria.fr/hal-01264233.

In addition, static and dynamic concerns aim to address multi-platform cross-usages, from large to embedded scales.

## Installation

```
$ make build
```

## Test

```
$ make test
```

## Main abstractions

The distclus library intend to be polymorphic :
any element type can be used provided a distance and a barycenter can be computed between 2 elements.

A data type that can be clustered must implement the ```core.Elemt``` interface.
Distance and barycenter are calculated using a suitable object that implements ```core.Space```.

An algorithm is represented by an ```core.Algo``` object.
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

Algorithms are configured with configuration objects.

All algorithm configurations extend the `core.Conf` :

```go
package core

type Conf struct {
	Iter           uint64
	IterFreq       float64
    Timeout        float64
    NumCPU         int
    DataPerIter   uint64
	StatusNotifier StatusNotifier
}
```

Where :

- `Iter`: maximal number of iterations if given. Unlimited by default.
- `IterFreq`: maximal number of iterations per second. Unlimited by default.
- `Timeout`: maximal algorithm execution duration in seconds. Unlimited by default.
- `NumCPU`: number of CPU to use for algorithm execution. Default is maximal number of CPU.
- `DataPerIter`: minimum number of pushed data before starting a new iteration if given. Online clustering specific.
- `StatusNotifier`: callback called in a separate go routine, each time the algorithm change of status or fires an error. Online clustering specific.

### MCMC Configuration

A minimal configuration for the MCMC algorithm requires
two objects, one for the Metropolis Hastings and the other for the alteration distribution:

```go
package main
import "distclus/mcmc"

var conf = mcmc.Conf{
	InitK: 1,
	Amp:   .5,
	B:     1,
    Conf:  core.Conf{
      Iter: 1000,
    }
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

For more information on setting these parameters refer to https://hal.inria.fr/hal-01264233.

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
func Build(conf mcmc.Conf, tConf mcmc.MultivTConf) (algo *core.Algo, space core.Space) {
	space = euclid.NewSpace(euclid.Conf{})
	var distrib = mcmc.NewMultivT(tConf) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, nil, kmeans.PPInitializer, distrib)
	return
}
```

## Run and feed

The algorithm can be run in two modes :
 - `batch` : all data must be pushed before starting the algorithm.
 - `online` : further data can be pushed after the algorithm is started, and dynamic functionalities are given for interacting continuously with the clustering

The following function starts the algorithm in online mode then pushes the observations

```go
package main
import (
	"distclus/core"
)

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
	for i := 0; i < len(observations) && err == nil; i++ {
		err = algo.Push(observations[i])
	}
	err = algo.Batch() // run the algorithm in batch mode
	return
}
```

## Prediction

Once the algorithm is started, either in batch or online mode,
the ```Predict``` method can be used to make predictions.
The following function uses predictions to calculate the root mean squared error
for observations for which real output is known.

```go
package main
import (
	"distclus/core"
	"math"
)

func RMSE(algo *core.Algo, observations []core.Elemt, output []core.Elemt, space core.Space) float64 {
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
)

func Eval(algo *core.Algo, centers core.Clust, observations []core.Elemt, space core.Space) (result core.Clust, rmse float64, err error) {
	var output = getOutput(centers, observations, space)
	rmse = RMSE(algo, observations, output, space)
	result, err = algo.Centroids()
	return
}

func getOutput(centers core.Clust, observations []core.Elemt, space core.Space) (output []core.Elemt) {
	var labels = centers.MapLabel(observations, space)
	output = make([]core.Elemt, len(labels))
	for i := range labels {
		output[i] = centers[labels[i]]
	}
	return
}
```

## Sample data

For testing purpose we need data.
The following function returns real centers with the given number of sample observations.

```go
package main
import (
	"distclus/core"
	"golang.org/x/exp/rand"
)

func Sample(n int) (centers core.Clust, observations []core.Elemt) {
	centers = core.Clust(
		[]core.Elemt{
			[]float64{1.4, 0.7},
			[]float64{7.6, 7.6},
		})
	observations = make([]core.Elemt, n)
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

## Putting all together

We will now glue the pieces we have built so far. The algorithm must be closed to stop online execution if necessary.
In the following example it is deferred to the end of the function.

```go
package main
import (
	"fmt"
	"time"
)

func Example() {
	var centers, observations = Sample(1000)
	var train, test = observations[:800], observations[800:]

	var algo, space = Build(conf, tConf)
	defer algo.Close()

	var errRun = RunAndFeed(algo, train)

	if errRun == nil {
		var result, rmse, errEval = Eval(algo, centers, test, space)
		fmt.Printf("%v %v %v\n", len(result) < 4, rmse < 1, errEval)
	}
	// Output: true true <nil>
}
```

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

### `core.OnlineClust` interface

The `core.OnlineClust` interface is implemented by the `core.Algo` struct.

We have covered almost all methods so far:

 - `Play() error`: execute the algorithm.
 - `Centroids() (Clust, error)`: get array of clustering centroids.
 - `Predict(elemt Elemt) (Elemt, int, error)`: according to previous method, get centroid and its index in array of clustering centroids for input elemt.
 - `Push(elemt Elemt) error`: push an element.
 - `Stop() error`: stop execution.
 - `Pause() error`: pause execution. The method `Play` goes back to execution.
 - `Wait() error`: wait until algorithm status equals Sleeping or ends the execution.
 - `Batch() error` execute the algo in batch mode. Similar to the call sequence of `Run`, `Wait` and `Stop`.
 - `Status() core.ClustStatus`: get algo status.
 - `Running() bool`: true iif algo is in running status (`core.Running`, `core.Idle` and `core.Sleeping`).

Algorithms may return specific figures that describes their running state. These can be obtained by the
`RuntimeFigures` method.
```go
var rt, err = algo.RuntimeFigures()
if err == nil {
  for name, value := range rt {
  	fmt.Printf("%v: %v\n", name, value)
  }
}
```

### ```mcmc.LateDistrib``` struct

The ```mcmc.MutlivT``` implements a multivariate T distribution. The dimension of the data must be known and set
in the ```mcmc.MultivTConf``` configuration object. In some situation this information is not known until the first
data arrives. This can be handled using a ```LateDistrib``` instance. It implements the ```Distrib``` interface and
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

The functor passed to `mcmc.NewLateDistrib` is executed only once with minimal locking to ensure parallel safety.

## Dynamic features

The algorithm is executed asynchronously and continuously, allowing new data to be pushed during execution.

This is achieved by calling the method `Play`, launching the algorithm execution as a separate go routine, and waiting for status equals `core.Running`.

When the algorithm starts, it first initializes the starting centers.
For example, the number of initial centroids is given by the parameter `InitK` of the `mcmc.Conf` configuration object (see above).
Thus at least `InitK` observations must be given at construction time or pushed before the algorithm starts,
otherwise an error is returned by the `Play` method.

In such mode, the parameters `Iter`, `DataPerIter` and `IterFreq` of the `core.Conf` are both used to temporize continuous execution by respectively execute `Iter` iterations after a last `DataPerIter` pushed data and ensure maximum number of iterations per seconds.

Remainding methods allow you to dynamically interact with the algorithm:
- `Play() error`: start the algorithm if not running (status `core.Created`, `core.Ready` or `core.Failed`), otherwise (status `core.Idle`), goes back to execution. Release when algorithm status equals `core.Running`.
- `Pause() error`: pause the algorithm. Wait until the algo is `core.Idle`.
- `Wait() error`: wait until the algorithm is `core.Sleeping`, `core.Ready` or `core.Failed` status.
- `Stop() error`: stop the algorithm and wait until it is ready. Such algorithm enters in `core.Stopping` before `core.Ready`.
- `Status() core.ClustStatus`: get algo status.
- `Running() bool`: true iif algo is in running status (`core.Running`, `core.Idle` and `core.Sleeping`).
- `StatusNotifier(core.ClustStatus, error)`: callback function when algo status change or an error is raised. Setted in `core.Conf.StatusNotifier`.

The `RunAndFeed` function above may be modified like this:

```go
package main
import (
	"distclus/core"
)

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
    for i := 0; i < conf.InitK && err == nil; i++ {
        err = algo.Push(observations[i])
    }
    if err != nil {
        err = algo.Play()
        for i := conf.initK; i < len(observations) && err == nil; i++ {
            err = algo.Push(observations[i])
        }
        algo.Wait()  // let the algorithm converge
    }
    return
}

```

In a real life situation of course this is not needed:
The online algorithm will be closed only when the service is shutdown and data will be pushed gradually when they arrive.

## More data types

In the example above the observations where vectors of R<sup>2</sup> and the distance used was the Euclid distance.
The data types and distance are defined by objects that implement the `core.Space` interface.

The library provides 3 different data types :
 - `euclid.Space` built with `euclid.NewSpace` constructor, used for vectors with Euclid distance
 - `cosinus.Space` built with `cosinus.NewSpace` constructor, used for vectors with cosinus distance
 - `dtw.Space` built with `dtw.NewSpace` constructor, used for time series of vectors with dtw distance

 ### Time series

 In order to manipulate time series instead of simple vectors,
 all the specific stuff has to be done at construction time.
 In our example above the `Build` function should be modified as follow :

 ```go
package main
import (
	"distclus/core"
	"distclus/euclid"
	"distclus/dtw"
	"distclus/kmeans"
	"distclus/mcmc"
)

func Build(conf mcmc.Conf) (algo *core.Algo, space core.Space) {
	var inner = euclid.NewSpace(euclid.Conf{})
	space = dtw.NewSpace(dtw.Conf{InnerSpace: inner})
	var distrib = mcmc.NewDirac()
	algo = mcmc.NewAlgo(conf, space, nil, kmeans.PPInitializer, distrib)
	return
}
```

The `dtw.Space` object uses internally an `euclid.Space`
but it could use another space as well (such as the cosinus space).

The library does not provide a distribution that handle time series,
the generic Dirac distribution (which actually does not alter the centers) can be used instead.

Of course the `Sample` function should be also modified to sample time series instead of vectors.

## More algorithms

We have seen above that 3 algorithms are provided with the library. In the example we saw the MCMC algorithm in depth.

### The streaming algorithm

The streaming algorithm requires few computing resources. It is especially suitable for online usage.
Once again the modification is done at build time. The following modification of the ```Build``` function build
a streaming algorithm for time series:

```go
package main
import (
	"distclus/core"
	"distclus/euclid"
	"distclus/dtw"
	"distclus/streaming"
)

func Build(conf streaming.Conf) (algo *core.Algo, space core.Space) {
	var inner = euclid.NewSpace(euclid.Conf{})
	space = dtw.NewSpace(dtw.Conf{InnerSpace: inner})
	algo = streaming.NewAlgo(conf, space, nil)
	return
}
```

## Add your own algorithm

You can start to create your own algorithm by copying the template package and inspirate from other packages.

### file description

- template/conf: Algorithm configuration properties definition.

- template/impl: Algorithm implementation definition.

- template/algo: Bind algorithm configuration, implementation and user space into a core.Algo structure.
