package mcmc_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/vectors"
	"reflect"
	"testing"
)

var ints = []core.Elemt{
	[]float64{7},
	[]float64{1},
	[]float64{1},
	[]float64{1},
	[]float64{38},
	[]float64{2},
	[]float64{1},
	[]float64{7},
	[]float64{2},
	[]float64{2},
}

func Test_DistinctValuesMaxK(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:    2,
		Amp:      100000,
		McmcIter: 2000,
		MaxK:     50,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, vectors.Space{}, []core.Elemt{}, initializer)

	for _, v := range ints {
		_ = algo.Push(v)
	}

	_ = algo.Run(false)

	var centroids, _ = algo.Centroids()

	if len(centroids) != 4 {
		t.Error("Expected 4 centroids")
	}

	for _, e := range ints {
		var pred, _, _ = algo.Predict(e)
		if !reflect.DeepEqual(pred, e) {
			t.Error("Center and element should be equal")
		}
	}
}
