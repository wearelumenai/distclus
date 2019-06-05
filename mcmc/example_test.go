package mcmc_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
	"fmt"
	"golang.org/x/exp/rand"
	"math"
	"time"
)

var conf = mcmc.Conf{
	InitK: 1,
	Amp:   1000,
	B:     .1,
}

var tConf = mcmc.MultivTConf{
	Dim: 2,
	Nu:  3,
}

func Example() {
	var centers, observations = Sample()
	var algo, space = Build()
	var errRun = RunAndFeed(algo, observations)

	if errRun == nil {
		var result, rmse, errEval = Eval(algo, centers, observations, space)
		fmt.Printf("%v %v %v\n", errEval, len(result) < 4, rmse < 1)
	}

	_ = algo.Close()
	// Output: <nil> true true
}

func Build() (algo *core.Algo, space euclid.Space) {
	space = euclid.NewSpace(euclid.Conf{})
	var distrib = mcmc.NewMultivT(tConf) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, nil, kmeans.PPInitializer, distrib)
	return
}

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
	err = Feed(algo, observations[:10])
	if err != nil {
		return
	}
	err = algo.Run(true) // run the algorithm in background
	if err != nil {
		return
	}
	err = Feed(algo, observations[10:])
	time.Sleep(300 * time.Millisecond) // let the background algorithm converge
	return
}

func Eval(algo *core.Algo, centers core.Clust, observations []core.Elemt, space euclid.Space) (result core.Clust, rmse float64, err error) {
	var labels = centers.MapLabel(observations, space)
	rmse = RMSE(algo, centers, labels, observations, space)
	result, err = algo.Centroids()
	return
}

func Feed(algo *core.Algo, observations []core.Elemt) (err error) {
	for _, obs := range observations {
		err = algo.Push(obs)
		if err != nil {
			return
		}
	}
	return
}

func Sample() (centers core.Clust, observations []core.Elemt) {
	centers = core.Clust(
		[]core.Elemt{
			[]float64{1.4, 1.2},
			[]float64{7.6, 7.6},
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

func RMSE(algo *core.Algo, centers core.Clust, labels []int, observations []core.Elemt, space euclid.Space) float64 {
	var mse = 0.
	for i, label := range labels {
		var prediction, _, _ = algo.Predict(observations[i])
		var dist = space.Dist(prediction, centers[label])
		mse += dist * dist / float64(len(labels))
	}
	return math.Sqrt(mse)
}
