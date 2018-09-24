package kmeans_test

import (
	"distclus/kmeans"
	"distclus/internal/test"
	"distclus/core"
	"distclus/real"
	"golang.org/x/exp/rand"
	"testing"
)

func TestKMeans_Initialization(t *testing.T) {
	var conf = kmeans.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, km)
}

func TestKMeans_RunSyncGiven(t *testing.T) {
	var conf = kmeans.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, km)
}

func TestKMeans_RunSyncKMeansPP(t *testing.T) {
	var seed = uint64(187236548914256543)
	rgen := rand.New(rand.NewSource(seed))
	var conf = kmeans.KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
	var km = kmeans.NewSeqKMeans(conf, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestRunSyncKMeansPP(t, km)
	test.DoTestRunSyncCentroids(t, km)
}

func TestKMeans_RunAsync(t *testing.T) {
	var conf = kmeans.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, km)
	test.DoTestRunAsyncCentroids(t, km)
}

func TestKMeans_Workflow(t *testing.T) {
	var conf = kmeans.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = kmeans.NewSeqKMeans(conf, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestWorkflow(t, km)
}

func TestKMeans_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = kmeans.KMeansConf{Iter: 1, K: 2, Space: real.RealSpace{}}
		var km = kmeans.NewSeqKMeans(conf, init, []core.Elemt{})

		return km
	}

	test.DoTestEmpty(t, builder)
}