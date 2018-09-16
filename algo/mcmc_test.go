package algo_test

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"distclus/algo/zetest"
)

var mcmcConf = algo.MCMCConf{
	Dim:      5, FrameSize: 8, B: 100, Amp: 1,
	Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: real.RealSpace{},
}

var distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})

func TestMCMC_Centroids(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTest_Centroids(t, &mcmc)
}

func TestMCMC_Run_Sync(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 100
	conf.InitK = 1
	var seed = uint64(1872365454256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Sync(t, &mcmc)
}

func TestMCMC_Predict_Given(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = mcmcConf
		conf.McmcIter = 0
		var mcmc = algo.NewMCMC(conf, distrib, init, nil)
		return &mcmc
	}

	zetest.DoTest_Predict_Given(t, builder)
}

func TestMCMC_Predict_KMeansPP(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = mcmcConf
		conf.ProbaK = []float64{1, 8, 1}
		var seed = uint64(187232542653256543)
		conf.RGen = rand.New(rand.NewSource(seed))
		var mcmc = algo.NewMCMC(conf, distrib, init, nil)
		return &mcmc
	}

	zetest.DoTest_Predict_KMeansPP(t, builder)
}

func TestMCMC_Run_Async(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Async(t, &mcmc)
}

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var mcmc = algo.NewMCMC(conf, distrib, algo.KmeansPPInitializer, nil)

	zetest.DoTest_Workflow(t, &mcmc)
}

func TestMCMC_MaxK(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var mcmc = algo.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	for _, elemt := range zetest.TestVectors {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)

	var clust, _ = mcmc.Centroids()

	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func TestMCMC_Conf(t *testing.T) {
	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	func() {
		defer testPanic()
		var conf = mcmcConf
		conf.McmcIter = -10
		algo.NewMCMC(conf, distrib, algo.KmeansPPInitializer, nil)
	}()

	func() {
		defer testPanic()
		var conf = mcmcConf
		conf.InitK = 3
		var mcmc = algo.NewMCMC(conf, distrib, algo.KmeansPPInitializer, nil)
		mcmc.Run(false)
	}()
}

