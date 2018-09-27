package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	real "distclus/real"
	"testing"
)

func TestKMeans_Initialization(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 0}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, km)
}

func TestKMeans_DefaultConf(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: core.AlgoConf{
		Space: real.RealSpace{},
		InitK: 3,
	}, Iter: 0}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestInitialization(t, km)
}


func TestKMeans_RunSyncGiven(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 0}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, km)
}

func TestKMeans_RunSyncKMeansPP(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 20}
	var km = kmeans.NewSeqKMeans(conf, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestRunSyncKMeansPP(t, km)
	test.DoTestRunSyncCentroids(t, km)
}

func TestKMeans_RunAsync(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 1 << 30}
	var km = kmeans.NewSeqKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, km)
	test.DoTestRunAsyncCentroids(t, km)
}

func TestKMeans_Workflow(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 1 << 30}
	var km = kmeans.NewSeqKMeans(conf, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestWorkflow(t, km)
}

func TestKMeans_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = kmeans.KMeansConf{AlgoConf: algoConf, Iter: 1}
		var km = kmeans.NewSeqKMeans(conf, init, []core.Elemt{})

		return km
	}

	test.DoTestEmpty(t, builder)
}
