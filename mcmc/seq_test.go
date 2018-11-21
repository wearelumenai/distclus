package mcmc_test

import (
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"testing"

	"golang.org/x/exp/rand"
)

var mcmcConf = mcmc.MCMCConf{
	AlgorithmConf: oc.AlgorithmConf{
		Space: real.RealSpace{},
	},
	InitK:     3,
	FrameSize: 8,
	RGen:      rand.New(rand.NewSource(6305689164243)),
	Dim:       5, B: 100, Amp: 1,
	Norm: 2, Nu: 3, McmcIter: 20,
	InitIter: 0,
}

var distrib = mcmc.NewMultivT(mcmc.MultivTConf{mcmcConf})

func TestMCMC_Initialization(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestInitialization(t, algo)
}

func TestMCMC_DefaultConf(t *testing.T) {
	var conf = mcmcConf
	conf.RGen = nil
	conf.McmcIter = 0
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestInitialization(t, algo)
}

func TestMCMC_RunSyncGiven(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestRunSyncGiven(t, algo)
}

func TestMCMC_RunSyncKMeansPP(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232592652256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []oc.Elemt{})

	test.DoTestRunSyncKMeansPP(t, algo)
}

func TestMCMC_RunAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestRunAsync(t, algo)
}

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.KMeansPPInitializer, []oc.Elemt{})

	test.DoTestWorkflow(t, algo)
}

func TestMCMC_MaxK(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 10
	conf.MaxK = 6
	conf.Amp = 1e6
	var algo = mcmc.NewSeqMCMC(conf, distrib, kmeans.GivenInitializer, []oc.Elemt{})

	test.PushAndRunSync(algo)

	var clust, _ = algo.Centroids()
	if l := len(clust); l > 6 {
		t.Error("Exepected ", conf.MaxK, "got", l)
	}
}

func TestMCMC_AcceptRatio(t *testing.T) {
	var algo = mcmc.NewSeqMCMC(mcmcConf, distrib, kmeans.GivenInitializer, []oc.Elemt{})
	test.PushAndRunSync(algo)
	var r = algo.AcceptRatio()
	if r < 0 || r > 1 {
		t.Error("Expected ratio in [0 1], got", r)
	}
}
