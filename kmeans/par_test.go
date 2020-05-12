package kmeans_test

import (
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/kmeans"
)

var space = euclid.Space{}

func Test_ParPredictGiven(t *testing.T) {
	kmeansConf := kmeans.Conf{K: 3, Par: true}
	var algo = kmeans.NewAlgo(kmeansConf, space, []core.Elemt{}, kmeans.GivenInitializer)

	test.DoTestInitGiven(t, algo)
}

func Test_ParRunSyncPP(t *testing.T) {
	kmeansConf := kmeans.Conf{K: 3, CtrlConf: core.CtrlConf{Iter: 20}, RGen: rgen(), Par: true}
	var algo = kmeans.NewAlgo(kmeansConf, space, []core.Elemt{}, kmeans.PPInitializer)

	test.DoTestRunSyncPP(t, algo)
	test.DoTestRunSyncCentroids(t, algo)
}
func Test_ParRunAsync(t *testing.T) {
	var implConf = kmeans.Conf{K: 3, CtrlConf: core.CtrlConf{Iter: 1000}, RGen: rgen(), Par: true}
	var initializer = kmeans.GivenInitializer
	var algo = kmeans.NewAlgo(implConf, space, []core.Elemt{}, initializer)

	test.DoTestRunAsync(t, algo)
	test.DoTestRunAsyncCentroids(t, algo)
	test.DoTestRunAsyncPush(t, algo)
}
