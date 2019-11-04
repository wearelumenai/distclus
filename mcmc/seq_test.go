package mcmc_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"testing"

	"golang.org/x/exp/rand"
)

var space = euclid.Space{}

func Test_Initialization(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 0.1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestInitialization(t, algo)
}

func Test_DefaultConf(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      nil,
		B:         100, Amp: 1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestInitialization(t, algo)
}

func Test_RunSyncGiven(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestRunSyncGiven(t, algo)
}

func Test_RunSyncKMeansPP(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		ProbaK:    []float64{1, 8, 1},
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 0.1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestRunSyncPP(t, algo)
}

func Test_RunAsync(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
		// FrameSize: 8,
		RGen: rand.New(rand.NewSource(6305689164243)),
		B:    100, Amp: 0.1,
		Norm:   2,
		ProbaK: []float64{1, 8, 1},
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestRunAsync(t, algo)
	test.DoTestRunAsyncPush(t, algo)
}

func Test_Workflow(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestWorkflow(t, algo)
}

func Test_MaxK(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1e6,
		Norm: 2,
		MaxK: 6,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.PushAndRunSync(algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", implConf.MaxK, "got", l)
	}
}

func Test_AcceptRatio(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.PushAndRunSync(algo)
	var rf, _ = algo.RuntimeFigures()
	var r = rf["acceptations"] / rf["iterations"]
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}

func Test_TimeOut(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	for _, elemt := range test.Vectors {
		_ = algo.Push(elemt)
	}
	var err = algo.Run()
	if err != core.ErrTimeOut {
		t.Error("time out expected")
	}
	_ = algo.Close()
}
