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
	kmeansConf := kmeans.Conf{K: 3, Iter: 0, Par: true}
	var conf = core.Conf{ImplConf: kmeansConf, SpaceConf: nil}
	var algo = kmeans.NewAlgo(conf, space, []core.Elemt{}, kmeans.GivenInitializer)

	test.DoTestRunSyncGiven(t, &algo)
}

func Test_ParRunSyncPP(t *testing.T) {
	kmeansConf := kmeans.Conf{K: 3, Iter: 20, RGen: rgen(), Par: true}
	var conf = core.Conf{ImplConf: kmeansConf, SpaceConf: nil}
	var algo = kmeans.NewAlgo(conf, space, []core.Elemt{}, kmeans.PPInitializer)

	test.DoTestRunSyncPP(t, &algo)
	test.DoTestRunSyncCentroids(t, &algo)
}

func Test_ParRunAsync(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, Iter: 1 << 30, RGen: rgen(), Par: true}
	var conf = core.Conf{
		ImplConf:  implConf,
		SpaceConf: nil,
	}
	var initializer = kmeans.GivenInitializer
	var algo = kmeans.NewAlgo(conf, space, []core.Elemt{}, initializer)

	test.DoTestRunAsync(t, &algo)
	test.DoTestRunAsyncCentroids(t, &algo)
	test.DoTestRunAsyncPush(t, &algo)
}
