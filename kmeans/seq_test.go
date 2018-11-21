package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"

	"golang.org/x/exp/rand"
)

func Test_Initialization(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 0}
	var km = kmeans.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, km)
}

func Test_DefaultConf(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 0}
	var km = kmeans.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, km)
}

func Test_RunSyncGiven(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 0}
	var km = kmeans.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, km)
}

func rgen() *rand.Rand {
	return rand.New(rand.NewSource(6305689164243))
}

func Test_RunSyncPP(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 20, RGen: rgen()}
	var km = kmeans.NewSeqImpl(conf, kmeans.PPInitializer, []core.Elemt{})

	test.DoTestRunSyncPP(t, km)
	test.DoTestRunSyncCentroids(t, km)
}

func Test_RunAsync(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 1 << 30, RGen: rgen()}
	var km = kmeans.NewSeqImpl(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, km)
	test.DoTestRunAsyncCentroids(t, km)
}

func Test_Workflow(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 1 << 30, RGen: rgen()}
	var km = kmeans.NewSeqImpl(conf, kmeans.PPInitializer, []core.Elemt{})

	test.DoTestWorkflow(t, km)
}

func Test_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 1, RGen: rgen()}
		var km = kmeans.NewSeqImpl(conf, init, []core.Elemt{})

		return km
	}

	test.DoTestEmpty(t, builder)
}
