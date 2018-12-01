package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"

	"golang.org/x/exp/rand"
)

func Test_Initialization(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 0}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestInitialization(t, &algo)
}

func Test_DefaultConf(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 0}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestInitialization(t, &algo)
}

func Test_RunSyncGiven(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 0}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncGiven(t, &algo)
}

func rgen() *rand.Rand {
	return rand.New(rand.NewSource(6305689164243))
}

func Test_RunSyncPP(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 20, RGen: rgen()}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.PPInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncPP(t, &algo)
	test.DoTestRunSyncCentroids(t, &algo)
}

func Test_RunAsync(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 1 << 30, RGen: rgen()}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncCentroids(t, &algo)
}

func Test_Workflow(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 1 << 30, RGen: rgen()}
	var impl = kmeans.NewSeqImpl(&conf, kmeans.PPInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestWorkflow(t, &algo)
}

func Test_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = kmeans.Conf{K: 3, Iter: 1, RGen: rgen()}
		var impl = kmeans.NewSeqImpl(&conf, init, []core.Elemt{})
		var algo = core.NewAlgo(conf, &impl, space)

		return &algo
	}

	test.DoTestEmpty(t, builder)
}
