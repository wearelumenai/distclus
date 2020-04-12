package mcmc_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/figures"
	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"

	"golang.org/x/exp/rand"
)

var space = euclid.Space{}

func Test_Initialization(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
		RGen:  rand.New(rand.NewSource(6305689164243)),
		B:     100, Amp: 0.1,
		Norm:      2,
		FrameSize: 8,
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
		InitK: 3,
		RGen:  nil,
		B:     100, Amp: 1,
		Norm:      2,
		FrameSize: 8,
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
		InitK: 3,
		RGen:  rand.New(rand.NewSource(6305689164243)),
		B:     100, Amp: 1,
		Norm:      2,
		FrameSize: 8,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestInitGiven(t, algo)
}

func Test_RunSyncKMeansPP(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:  3,
		ProbaK: []float64{1, 8, 1},
		RGen:   rand.New(rand.NewSource(6305689164243)),
		B:      100, Amp: 0.1,
		Norm:      2,
		FrameSize: 8,
		Conf:      core.Conf{Iter: 1},
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
		Conf:   core.Conf{Iter: 100},
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
		InitK: 3,
		RGen:  rand.New(rand.NewSource(6305689164243)),
		B:     100, Amp: 1,
		Norm:      2,
		FrameSize: 8,
		Conf:      core.Conf{Iter: 20},
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
		InitK: 3,
		RGen:  rand.New(rand.NewSource(6305689164243)),
		B:     100, Amp: 1e6,
		Norm:      2,
		MaxK:      6,
		FrameSize: 8,
		Conf:      core.Conf{Iter: 10},
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
		InitK: 3,
		RGen:  rand.New(rand.NewSource(6305689164243)),
		B:     100, Amp: 1,
		Norm:      2,
		FrameSize: 8,
		Conf:      core.Conf{Iter: 20},
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
	var r = rf[figures.Acceptations] / rf[figures.Iterations]
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}
