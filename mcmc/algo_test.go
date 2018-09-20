package mcmc_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/zetest"
	"distclus/real"
	"golang.org/x/exp/rand"
	"testing"
)

var mcmcConf = mcmc.MCMCConf{
	Dim: 5, FrameSize: 8, B: 100, Amp: 1,
	Norm: 2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: real.RealSpace{},
}

var distrib = mcmc.NewMultivT(mcmc.MultivTConf{mcmcConf})

func TestMCMC_Initialization(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	zetest.DoTestInitialization(t, algo)
}

func TestMCMC_RunSyncGiven(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	zetest.DoTestRunSyncGiven(t, algo)
}

func TestMCMC_RunSyncKMeansPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232542653256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})

	zetest.DoTestRunSyncKMeansPP(t, algo)
}

func TestMCMC_RunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	zetest.DoTestRunAsync(t, algo)
}

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})

	zetest.DoTestWorkflow(t, algo)
}

func TestMCMC_MaxK(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []core.Elemt{})

	zetest.PushAndRunSync(algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func TestMCMC_AcceptRatio(t *testing.T) {
	var algo = mcmc.NewSeqMCMC(mcmcConf, distrib, kmeans.GivenInitializer, []core.Elemt{})
	zetest.PushAndRunSync(algo)
	var r = algo.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}

func TestMCMC_ConfErrorIter(t *testing.T) {
	defer zetest.AssertPanic(t)
	var conf = mcmcConf
	conf.McmcIter = -10
	mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})
}

func TestMCMC_ConfErrorK(t *testing.T) {
	defer zetest.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 3
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []core.Elemt{})
	algo.Run(false)
}
