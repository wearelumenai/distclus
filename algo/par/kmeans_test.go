package par_test

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"distclus/algo/par"
	"distclus/algo/zetest"
)

func TestKMeans_ParRun_Sync(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1, K: 3, Space: real.RealSpace{}}
	var km = par.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Sync(t, &km)
}

func TestKMeans_ParPredict_Given(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = algo.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
		var km = par.NewKMeans(conf, init, nil)
		return &km
	}

	zetest.DoTest_Predict_Given(t, builder)
}

func TestKMeans_ParPredict_KMeansPP(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var seed = uint64(187236548914256543)
		rgen := rand.New(rand.NewSource(seed))
		var conf = algo.KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
		var km = par.NewKMeans(conf, algo.KmeansPPInitializer, nil)
		return &km
	}

	var km = zetest.DoTest_Predict_KMeansPP(t, builder)
	zetest.DoTest_Predict_Centroids(t, km.(*algo.KMeans))
}

func TestKMeans_ParRUN_Async(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = par.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Async(t, &km)
	zetest.DoTest_Run_Async_Centroids(t, &km)
}
