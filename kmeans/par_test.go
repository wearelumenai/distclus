package kmeans_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/vectors"
	"testing"
)

var space = vectors.Space{}

func Test_ParPredictGiven(t *testing.T) {
	kmeansConf := kmeans.Conf{K: 3, Iter: 0}
	var conf = core.Conf{ImplConf: kmeansConf, SpaceConf: nil}
	var impl = kmeans.NewParImpl(&kmeansConf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncGiven(t, &algo)
}

func Test_ParRunSyncPP(t *testing.T) {
	kmeansConf := kmeans.Conf{K: 3, Iter: 20, RGen: rgen()}
	var conf = core.Conf{ImplConf: kmeansConf, SpaceConf: nil}
	var impl = kmeans.NewParImpl(&kmeansConf, kmeans.PPInitializer, []core.Elemt{})
	var algo = core.NewAlgo(conf, &impl, space)

	test.DoTestRunSyncPP(t, &algo)
	test.DoTestRunSyncCentroids(t, &algo)
}

func Test_ParRunAsync(t *testing.T) {
	var conf = kmeans.Conf{K: 3, Iter: 1 << 30, RGen: rgen()}
	var impl = kmeans.NewParImpl(&conf, kmeans.GivenInitializer, []core.Elemt{})
	var algo = core.NewAlgo(core.Conf{ImplConf: conf, SpaceConf: nil}, &impl, space)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncCentroids(t, &algo)
}
