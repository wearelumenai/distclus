package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"testing"

	"golang.org/x/exp/rand"
)

var _mcmcConf = mcmc.Conf{
	InitK:     3,
	FrameSize: 8,
	RGen:      rand.New(rand.NewSource(6305689164243)),
	Dim:       5, B: 100, Amp: 1,
	Norm: 2, Nu: 3, McmcIter: 20,
	InitIter: 0,
}

var distrib = mcmc.NewMultivT(mcmc.MultivTConf{_mcmcConf})
var space = real.Space{}

func Test_Initialization(t *testing.T) {
	var conf = _mcmcConf
	conf.McmcIter = 0
	var impl = mcmc.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestInitialization(t, &algo)
}

func Test_DefaultConf(t *testing.T) {
	var conf = _mcmcConf
	conf.RGen = nil
	conf.McmcIter = 0
	var impl = mcmc.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestInitialization(t, &algo)
}

func Test_RunSyncGiven(t *testing.T) {
	var conf = _mcmcConf
	conf.McmcIter = 0
	var impl = mcmc.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncGiven(t, &algo)
}

func Test_RunSyncKMeansPP(t *testing.T) {
	var conf = _mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232592652256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var impl = mcmc.NewSeqImpl(conf, kmeans.PPInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncPP(t, &algo)
}

func Test_RunAsync(t *testing.T) {
	var conf = _mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var impl = mcmc.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunAsync(t, &algo)
}

func Test_Workflow(t *testing.T) {
	var conf = _mcmcConf
	conf.McmcIter = 1 << 30
	var impl = mcmc.NewSeqImpl(conf, kmeans.PPInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestWorkflow(t, &algo)
}

func Test_MaxK(t *testing.T) {
	var conf = _mcmcConf
	conf.McmcIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var impl = mcmc.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

	test.PushAndRunSync(&algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func Test_AcceptRatio(t *testing.T) {
	var impl = mcmc.NewSeqImpl(_mcmcConf, kmeans.GivenInitializer, []core.Elemt{}, distrib)
	var algo = core.NewAlgo(_mcmcConf, &impl, space)
	test.PushAndRunSync(&algo)
	var r = impl.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}
