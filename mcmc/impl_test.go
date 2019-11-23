package mcmc_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
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
		InitK: 2,
		Amp:   1000000,
		B:     .00001,
		MaxK:  50,
		Conf:  core.Conf{Iter: 2000},
	}
	var tConf = mcmc.MultivTConf{
		Dim: 1,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, euclid.Space{}, []core.Elemt{}, initializer, distrib)

	for _, v := range ints {
		algo.Push(v)
	}

	algo.Batch()

	var centroids, _ = algo.Centroids()

	if len(centroids) != 4 {
		t.Error("Expected 4 centroids", len(centroids))
	}

	for _, e := range ints {
		var pred, _, _ = algo.Predict(e)
		if !reflect.DeepEqual(pred, e) {
			t.Error("Center and element should be equal")
		}
	}
}
