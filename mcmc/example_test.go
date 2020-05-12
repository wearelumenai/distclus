package mcmc_test

import (
	"fmt"
	"math"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"

	"golang.org/x/exp/rand"
)

var conf = mcmc.Conf{
	InitK:    1,
	Amp:      .5,
	B:        1,
	CtrlConf: core.CtrlConf{Iter: 20},
}

var tConf = mcmc.MultivTConf{
	Dim: 2,
	Nu:  3,
}

func Example() {
	var centers, observations = Sample(1000)
	var train, test = observations[:800], observations[800:]
	var algo, space = Build(conf, tConf)
	defer algo.Stop()

	var errRun = RunAndFeed(algo, train)

	if errRun == nil {
		var result, rmse, errEval = Eval(algo, centers, test, space)
		fmt.Printf("%v %v %v\n", len(result) < 4, rmse < 1, errEval)
	}
	// Output: true true <nil>
}

func Build(conf mcmc.Conf, tConf mcmc.MultivTConf) (algo *core.Algo, space core.Space) {
	space = euclid.NewSpace()
	var distrib = mcmc.NewMultivT(tConf) // the alteration distribution
	algo = mcmc.NewAlgo(conf, space, nil, kmeans.PPInitializer, distrib)
	return
}

func RunAndFeed(algo *core.Algo, observations []core.Elemt) (err error) {
	for i := 0; i < len(observations) && err == nil; i++ {
		err = algo.Push(observations[i])
	}
	err = algo.Batch(nil, 0)
	return
}

func Eval(algo *core.Algo, centers core.Clust, observations []core.Elemt, space core.Space) (result core.Clust, rmse float64, err error) {
	var output = getOutput(centers, observations, space)
	rmse = RMSE(algo, observations, output, space)
	result = algo.Centroids()
	return
}

func getOutput(centers core.Clust, observations []core.Elemt, space core.Space) (output []core.Elemt) {
	var labels, _ = centers.MapLabel(observations, space)
	output = make([]core.Elemt, len(labels))
	for i := range labels {
		output[i] = centers[labels[i]]
	}
	return
}

func RMSE(algo *core.Algo, observations []core.Elemt, output []core.Elemt, space core.Space) float64 {
	var mse = 0.
	for i := range observations {
		var _, _, dist = algo.Predict(observations[i])
		mse += dist * dist / float64(len(observations))
	}
	return math.Sqrt(mse)
}

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
