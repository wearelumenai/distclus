package mcmc_test

import (
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/zetest"
	"golang.org/x/exp/rand"
	"math"
	"testing"
)

func TestMCMC_ParPredict_Given(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewParMCMC(conf, distrib, kmeans.GivenInitializer, nil)

	zetest.DoTestRunSyncGiven(t, algo)
}

func TestMCMC_ParPredictKMeansPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232542653256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var algo = mcmc.NewParMCMC(conf, distrib, kmeans.KMeansPPInitializer, nil)

	zetest.DoTestRunSyncKMeansPP(t, algo)
}

func TestMCMC_ParRunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var algo = mcmc.NewParMCMC(conf, distrib, kmeans.GivenInitializer, nil)

	zetest.DoTestRunAsync(t, algo)
}

func TestParMCMCSupport_ParLoss(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewParMCMC(conf, distrib, kmeans.GivenInitializer, nil)

	zetest.PushAndRunSync(algo)

	var clust, _ = algo.Centroids()
	var l1 = algo.Loss(clust)
	var l2 = clust.Loss(algo.Data, algo.Space, algo.Norm)

	if math.Abs(l1-l2)>1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}
