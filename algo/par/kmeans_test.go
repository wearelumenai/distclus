package par_test

import (
	"distclus/real"
	"testing"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"distclus/algo/par"
	"distclus/algo/zetest"
)

func TestKMeans_ParPredictGiven(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
	var km = par.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTestRunSyncGiven(t, &km)
}

func TestKMeans_ParRunSyncKMeansPP(t *testing.T) {
	var seed = uint64(187236548914256543)
	rgen := rand.New(rand.NewSource(seed))
	var conf = algo.KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
	var km = par.NewKMeans(conf, algo.KMeansPPInitializer, nil)

	zetest.DoTestRunSyncKMeansPP(t, &km)
	zetest.DoTestRunSyncCentroids(t, &km)
}

func TestKMeans_ParRunAsync(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = par.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTestRunAsync(t, &km)
	zetest.DoTestRunAsyncCentroids(t, &km)
}
