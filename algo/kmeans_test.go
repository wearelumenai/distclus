package algo_test

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"distclus/algo/zetest"
)

func TestKMeans_Initialization(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 0, K: 4, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Initialization(t, &km)
}

func TestKMeans_Run_Sync(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1, K: 3, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Run_Sync(t, &km)
}

func TestKMeans_Predict_Given(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
	var km = algo.NewKMeans(conf, algo.GivenInitializer, nil)

	zetest.DoTest_Predict_Given(t, &km)
}

func TestKMeans_Predict_KmeansPP(t *testing.T) {
		var seed = uint64(187236548914256543)
		rgen := rand.New(rand.NewSource(seed))
		var conf = algo.KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
		var km = algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)

	zetest.DoTest_Predict_KmeansPP(t, &km)
	zetest.DoTest_Predict_Centroids(t, &km)
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

func TestKMeans_ConfError_Iter(t *testing.T) {
	func() {
		defer zetest.TestPanic(t)
		var conf = algo.KMeansConf{Iter: -10, K: 3, Space: real.RealSpace{}}
		var km = algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)
		km.Run(false)
	}()
}

func TestKMeans_ConfError_K(t *testing.T) {
	func() {
		defer zetest.TestPanic(t)
		var conf = algo.KMeansConf{Iter: 10, K: -3, Space: real.RealSpace{}}
		algo.NewKMeans(conf, algo.KmeansPPInitializer, nil)
	}()
}

