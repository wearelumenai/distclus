package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/vectors"
	"testing"

	"golang.org/x/exp/rand"
)

var space = vectors.Space{}

func Test_Initialization(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       5, B: 100, Amp: 0.1,
		Norm: 2, Nu: 3, McmcIter: 0,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestInitialization(t, &algo)
}

func Test_DefaultConf(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      nil,
		Dim:       5, B: 100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 0,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestInitialization(t, &algo)
}

func Test_RunSyncGiven(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       5, B: 100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 0,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncGiven(t, &algo)
}

func Test_RunSyncKMeansPP(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		ProbaK:    []float64{1, 8, 1},
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 0.1,
		Norm: 2, Nu: 3, McmcIter: 1,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncPP(t, &algo)
}

func Test_RunAsync(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
		// FrameSize: 8,
		RGen: rand.New(rand.NewSource(6305689164243)),
		Dim:  5, B: 100, Amp: 0.1,
		Norm: 2, Nu: 3, McmcIter: 20,
		InitIter: 0,
		ProbaK:   []float64{1, 8, 1},
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncPush(t, &algo)
}

func Test_Workflow(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       5, B: 100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 20,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestWorkflow(t, &algo)
}

func Test_MaxK(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       5, B: 100, Amp: 1e6,
		Norm: 2, Nu: 3, McmcIter: 10,
		MaxK:     6,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.PushAndRunSync(&algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.ImplConf.(mcmc.Conf).MaxK, "got", l)
	}
}

func Test_AcceptRatio(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		Dim:       5, B: 100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 20,
		InitIter: 0,
	}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.PushAndRunSync(&algo)
	var r = algo.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}
