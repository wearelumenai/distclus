package kmeans_test

import (
	"github.com/wearelumenai/distclus/v0/internal/test"
	"github.com/wearelumenai/distclus/v0/pkg/core"
	"github.com/wearelumenai/distclus/v0/pkg/kmeans"
	"testing"

	"golang.org/x/exp/rand"
)

func Test_Initialization(t *testing.T) {
	var implConf = kmeans.Conf{K: 3}
	var initializer = kmeans.GivenInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestInitialization(t, algo)
}

func Test_RunSyncGiven(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, Conf: core.Conf{Iter: 1}}
	var initializer = kmeans.GivenInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncGiven(t, algo)
}

func rgen() *rand.Rand {
	return rand.New(rand.NewSource(6305689164243))
}

func Test_RunSyncPP(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, Conf: core.Conf{Iter: 20}, RGen: rgen()}
	var initializer = kmeans.PPInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestRunSyncPP(t, algo)
	test.DoTestRunSyncCentroids(t, algo)
}

func Test_RunAsync(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, Conf: core.Conf{Iter: 1000}, RGen: rgen()}
	var initializer = kmeans.GivenInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestRunAsync(t, algo)
	test.DoTestRunAsyncCentroids(t, algo)
	test.DoTestRunAsyncPush(t, algo)
}

func Test_Workflow(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, Conf: core.Conf{Iter: 1000}, RGen: rgen()}
	var initializer = kmeans.PPInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestWorkflow(t, algo)
}

func Test_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var implConf = kmeans.Conf{K: 3, Conf: core.Conf{Iter: 1}, RGen: rgen()}
		var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, init)

		return algo
	}

	test.DoTestEmpty(t, builder)
}
