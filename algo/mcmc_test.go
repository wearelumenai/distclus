package algo_test

import (
	"distclus/algo"
	"distclus/algo/zetest"
	"distclus/real"
	"golang.org/x/exp/rand"
	"testing"
)

var mcmcConf = algo.MCMCConf{
	Dim: 5, FrameSize: 8, B: 100, Amp: 1,
	Norm: 2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: real.RealSpace{},
}

var distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})

func TestMCMC_Initialization(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTestInitialization(t, &mcmc)
}

func TestMCMC_RunSyncGiven(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTestRunSyncGiven(t, &mcmc)
}

func TestMCMC_RunSyncKMeansPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232542653256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = algo.NewMCMC(conf, distrib, algo.KMeansPPInitializer, nil)

	zetest.DoTestRunSyncKMeansPP(t, &mcmc)
}

func TestMCMC_RunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTestRunAsync(t, &mcmc)
}

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var mcmc = algo.NewMCMC(conf, distrib, algo.KMeansPPInitializer, nil)

	zetest.DoTestWorkflow(t, &mcmc)
}

func TestMCMC_MaxK(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.PushAndRunSync(&mcmc)

	var clust, _ = mcmc.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func TestMCMC_AcceptRatio(t *testing.T) {
	var mcmc = algo.NewMCMC(mcmcConf, distrib, algo.GivenInitializer, nil)
	zetest.PushAndRunSync(&mcmc)
	var r = mcmc.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}

func TestMCMC_ConfErrorIter(t *testing.T) {
	defer zetest.AssertPanic(t)
	var conf = mcmcConf
	conf.McmcIter = -10
	algo.NewMCMC(conf, distrib, algo.KMeansPPInitializer, nil)
}

func TestMCMC_ConfErrorK(t *testing.T) {
	defer zetest.AssertPanic(t)
	var conf = mcmcConf
	conf.InitK = 3
	var mcmc = algo.NewMCMC(conf, distrib, algo.KMeansPPInitializer, nil)
	mcmc.Run(false)
}
