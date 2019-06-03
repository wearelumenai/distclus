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

const Forever = int(^uint(0) >> 1)

var conf = mcmc.Conf{
	InitK:    2,
	Amp:      1.,
	Dim:      2,
	Nu:       3,
	McmcIter: Forever,
	RGen:     rand.New(rand.NewSource(uint64(time.Now().UnixNano()))),
}

func Example() {
	var centers, observations = Sample()
	var labels = centers.MapLabel(observations, space)

	var algo, space = buildAlgo()

	RunAndFeed(algo, observations)
	time.Sleep(time.Second)

	var rmse = RMSE(algo, centers, labels, observations, space)
	var result, _ = algo.Centroids()

	fmt.Println(len(result) < 4)
	fmt.Println(rmse < 1)
	// Output: true
	// true

	_ = algo.Close()
}

func RunAndFeed(algo *core.Algo, observations []core.Elemt) {
	for _, obs := range observations[:10] {
		_ = algo.Push(obs)
	}
	_ = algo.Run(true)
	for _, obs := range observations[10:] {
		_ = algo.Push(obs)
	}
}

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
			obs[j] += rand.Float64()*2 - 1
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

func buildAlgo() (*core.Algo, euclid.Space) {
	var distrib = mcmc.NewMultivT(mcmc.MultivTConf{conf})
	var space = euclid.NewSpace(euclid.Conf{})
	var algo = mcmc.NewAlgo(conf, space, nil, kmeans.PPInitializer, distrib)
	return algo, space
}
