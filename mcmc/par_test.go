package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"math"
	"runtime"
	"testing"

	"golang.org/x/exp/rand"
)

func Test_ParPredict_Given(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       3, B: 100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 0,
		InitIter: 1,
		Par:      true,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncGiven(t, &algo)
}

func Test_ParPredictPP(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		ProbaK:    []float64{1, 8, 1},
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       0, B: 100, Amp: 0.1,
		Norm: 2, Nu: 3, McmcIter: 20,
		InitIter: 0, Par: true,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncPP(t, &algo)
}

func Test_ParRunAsync(t *testing.T) {
	var conf = core.Conf{
		ImplConf: mcmc.Conf{
			InitK:     3,
			FrameSize: 8,
			ProbaK:    []float64{1, 8, 1},
			RGen:      rand.New(rand.NewSource(6305689164243)),
			B:         100, Amp: 0.1,
			Nu: 3, McmcIter: 5,
			InitIter: 1,
			Par:      true,
		},
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncPush(t, &algo)
}

func TestParStrategy_Loss(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 0,
		InitIter: 1,
		Par:      true,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.PushAndRunSync(&algo)

	var strategy = mcmc.ParStrategy{}
	buffer := core.NewDataBuffer(test.TestVectors, implConf.FrameSize)
	strategy.Degree = runtime.NumCPU()

	var clust, _ = algo.Centroids()
	var l1 = strategy.Loss(implConf, algo.Space(), clust, buffer.Data())
	var l2 = clust.Loss(test.TestVectors, algo.Space(), implConf.Norm)

	if math.Abs(l1-l2) > 1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}
