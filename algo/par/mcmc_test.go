package par_test

import (
	"testing"
	"distclus/real"
	"distclus/algo"
	"distclus/algo/par"
	"distclus/algo/zetest"
	"golang.org/x/exp/rand"
	"math"
)

var mcmcConf = algo.MCMCConf{
	Dim:      5, FrameSize: 8, B: 100, Amp: 1,
	Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: real.RealSpace{},
}

var distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})

func TestMCMC_ParPredict_Given(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = par.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTestRunSyncGiven(t, &mcmc)
}

func TestMCMC_ParPredictKMeansPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232542653256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = par.NewMCMC(conf, distrib, algo.KMeansPPInitializer, nil)

	zetest.DoTestRunSyncKMeansPP(t, &mcmc)
}

func TestMCMC_ParRunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var mcmc = par.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.DoTestRunAsync(t, &mcmc)
}

func TestParMCMCSupport_ParLoss(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = par.NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	zetest.PushAndRunSync(&mcmc)

	var clust, _ = mcmc.Centroids()
	var l1 = mcmc.Loss(mcmc, clust)
	var l2 = clust.Loss(mcmc.Data, mcmc.Space, mcmc.Norm)

	if math.Abs(l1-l2)>1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}
