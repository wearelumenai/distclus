package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"
)

func TestKMeans_ParPredictGiven(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgorithmConf: algoConf, K: 3, Iter: 0}
	var km = kmeans.NewParKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunSyncGiven(t, km)
}

func TestKMeans_ParRunSyncKMeansPP(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgorithmConf: algoConf, K: 3, Iter: 20, RGen: rgen()}
	var km = kmeans.NewParKMeans(conf, kmeans.KMeansPPInitializer, []core.Elemt{})

	test.DoTestRunSyncKMeansPP(t, km)
	test.DoTestRunSyncCentroids(t, km)
}

func TestKMeans_ParRunAsync(t *testing.T) {
	var conf = kmeans.KMeansConf{AlgorithmConf: algoConf, K: 3, Iter: 1 << 30, RGen: rgen()}
	var km = kmeans.NewParKMeans(conf, kmeans.GivenInitializer, []core.Elemt{})

	test.DoTestRunAsync(t, km)
	test.DoTestRunAsyncCentroids(t, km)
}
