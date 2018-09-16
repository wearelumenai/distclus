package algo_test

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"distclus/algo/zetest"
)

func TestKMeans_Centroids(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 0, K: 4, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Centroids(t, &km)
}

func TestKMeans_Run_Sync(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1, K: 3, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Sync(t, &km)
}

func TestKMeans_Predict_Given(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = algo.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
		var km = algo.NewKMeans(conf, init, nil)
		return &km
	}

	zetest.DoTest_Predict_Given(t, builder)
}

func TestKMeans_Predict_KMeansPP(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var seed = uint64(187236548914256543)
		rgen := rand.New(rand.NewSource(seed))
		var conf = algo.KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
		var km = algo.NewKMeans(conf, init, nil)
		return &km
	}

	var km = zetest.DoTest_Predict_KMeansPP(t, builder)
	zetest.DoTest_Predict_Centroids(t, km.(*algo.KMeans))
}

func TestKMeans_Run_Async(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Async(t, &km)
	zetest.DoTest_Run_Async_Centroids(t, &km)
}

func TestKMeans_Workflow(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)

	zetest.DoTest_Workflow(t, &km)
}

func TestKMeans_Empty(t *testing.T) {
	var builder = func(init core.Initializer) core.OnlineClust {
		var conf = algo.KMeansConf{Iter: 1, K: 2, Space: real.RealSpace{}}
		var km = algo.NewKMeans(conf, init, nil)

		return &km
	}

	zetest.DoTest_Empty(t, builder)
}

func TestKMeans_Conf(t *testing.T) {
	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	func() {
		defer testPanic()
		var conf = algo.KMeansConf{Iter: -10, K: 3, Space: real.RealSpace{}}
		var km = algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)
		km.Run(false)
	}()

	func() {
		defer testPanic()
		var conf = algo.KMeansConf{Iter: 10, K: -3, Space: real.RealSpace{}}
		algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)
	}()
}

