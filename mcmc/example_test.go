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

func Build(conf mcmc.Conf, tConf mcmc.MultivTConf, data []core.Elemt) (algo *core.Algo, space euclid.Space) {
	space = euclid.NewSpace(euclid.Conf{})
	var distrib = mcmc.NewMultivT(tConf) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, data, kmeans.PPInitializer, distrib)
	return
}

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
	err = algo.Run(true) // run the algorithm in background
	for i := 0; i < len(observations) && err == nil; i++ {
		err = algo.Push(observations[i])
	}
	return
}

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

func RMSE(algo *core.Algo, observations []core.Elemt, output []core.Elemt, space euclid.Space) float64 {
	var mse = 0.
	for i := range observations {
		var prediction, _, _ = algo.Predict(observations[i])
		var dist = space.Dist(prediction, output[i])
		mse += dist * dist / float64(len(observations))
	}
	return math.Sqrt(mse)
}

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
