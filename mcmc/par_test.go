package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"math"
	"runtime"
	"testing"

	"golang.org/x/exp/rand"
)

func Test_ParPredict_Given(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var impl = mcmc.NewParImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)
	test.DoTestRunSyncGiven(t, &algo)
}

func Test_ParPredictPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232592652256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var impl = mcmc.NewParImpl(conf, kmeans.PPInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)
	test.DoTestRunSyncPP(t, &algo)
}

func Test_ParRunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var impl = mcmc.NewParImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunAsync(t, &algo)
}

func TestParStrategy_Loss(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var impl = mcmc.NewParImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, real.Space{})

	test.PushAndRunSync(&algo)

	var strategy = mcmc.ParStrategy{}
	buffer := core.NewDataBuffer(test.TestVectors, conf.FrameSize)
	strategy.Degree = runtime.NumCPU()

	var clust, _ = algo.Centroids()
	var l1 = strategy.Loss(conf, algo.Space, clust, buffer)
	var l2 = clust.Loss(test.TestVectors, algo.Space, conf.Norm)

	if math.Abs(l1-l2) > 1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}
