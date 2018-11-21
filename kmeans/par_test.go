package kmeans_test

import (
	"distclus/internal/test"
	"distclus/kmeans"
	"testing"
)

func TestKMeans_ParPredictGiven(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 0}
	var km = kmeans.NewParImpl(conf, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestRunSyncGiven(t, km)
}

func TestKMeans_ParRunSyncKMeansPP(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 20, RGen: rgen()}
	var km = kmeans.NewParImpl(conf, kmeans.PPInitializer, []oc.Elemt{})

	test.DoTestRunSyncKMeansPP(t, km)
	test.DoTestRunSyncCentroids(t, km)
}

func TestKMeans_ParRunAsync(t *testing.T) {
	var conf = kmeans.Conf{AlgorithmConf: algoConf, K: 3, Iter: 1 << 30, RGen: rgen()}
	var km = kmeans.NewParImpl(conf, kmeans.GivenInitializer, []oc.Elemt{})

	test.DoTestRunAsync(t, km)
	test.DoTestRunAsyncCentroids(t, km)
}
