package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/real"
	"testing"
)

var space = real.Space{}

func TestKMeans_ParPredictGiven(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 0}
	var impl = kmeans.NewParImpl(conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncGiven(t, &algo)
}

func TestKMeans_ParRunSyncKMeansPP(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 20, RGen: rgen()}
	var impl = kmeans.NewParImpl(conf, kmeans.PPInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncPP(t, &algo)
	test.DoTestRunSyncCentroids(t, &algo)
}

func TestKMeans_ParRunAsync(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 1 << 30, RGen: rgen()}
	var impl = kmeans.NewParImpl(conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncCentroids(t, &algo)
}
