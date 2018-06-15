package algo

import (
	"distclus/core"
	"testing"
	"reflect"
	"time"
	"golang.org/x/exp/rand"
)

var data = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

func TestKMeans_Centroids(t *testing.T) {
	var conf = KMeansConf{Iter: 0, K: 4, Space: core.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < conf.K; i++ {
		if !reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected", data[i], "got", clust[i])
		}
	}

	km.Close()
}

func TestKMeans_Run(t *testing.T) {
	var conf = KMeansConf{Iter: 1, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < conf.K; i++ {
		if reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected average got", clust[i])
		}
	}

	km.Close()
}

func TestKMeans_Predict(t *testing.T) {
	var conf = KMeansConf{Iter: 0, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)

	for i, elemt := range data {
		var c, idx, _ = km.Predict(elemt, false)

		if i == 0 || i == 3 || i == 4 {
			if idx != 0 || !reflect.DeepEqual(c, data[0]) {
				t.Error("Expected center 0")
			}
		}

		if i == 1 || i == 5 {
			if idx != 1 || !reflect.DeepEqual(c, data[1]) {
				t.Error("Expected center 1")
			}
		}

		if i == 2 || i == 6 || i == 7 {
			if idx != 2 || !reflect.DeepEqual(c, data[2]) {
				t.Error("Expected center 2")
			}
		}
	}

	km.Close()
}

func TestKMeans_Predict2(t *testing.T) {
	var seed = uint64(187236548914256543)
	rgen := rand.New(rand.NewSource(seed))
	var conf = KMeansConf{Iter: 20, K: 3, Space: core.RealSpace{}, RGen: rgen}
	var km = NewKMeans(conf, KmeansPPInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = km.Predict(data[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}
	}

	for i := 3; i < len(data); i++ {
		var c, idx, _ = km.Predict(data[i], false)

		if i == 3 || i == 4 {
			if idx != iclust[0] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 0")
			}
		}

		if i == 5 {
			if idx != iclust[1] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 1")
			}
		}

		if i == 6 || i == 7 {
			if idx != iclust[2] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 2")
			}
		}
	}

	km.Close()
}

func TestKMeans_Close(t *testing.T) {
	var conf = KMeansConf{Iter: 1 << 30, K: 4, Space: core.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	if km.status != Created {
		t.Error("Expected status", Created, "got", km.status)
	}

	km.Run(true)

	if km.status != Running {
		t.Error("Expected status", Running, "got", km.status)
	}

	km.Close()

	if km.status != Closed {
		t.Error("Expected status", Closed, "got", km.status)
	}
}

func TestKMeans_Async(t *testing.T) {
	var conf = KMeansConf{Iter: 1 << 30, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, KmeansPPInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(true)

	time.Sleep(300 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, ix, _ = km.Predict(obs, true)

	time.Sleep(300 * time.Millisecond)
	var clust, _ = km.Centroids()

	if reflect.DeepEqual(clust[ix], c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = km.Predict(data[1], false)
	var _, ix5, _ = km.Predict(data[5], false)

	if ix != ix5 || ix != ix1 {
		t.Error("Expected same center")
	}

	km.Close()
}

func TestKMeans_Workflow(t *testing.T) {
	var conf = KMeansConf{Iter: 1 << 30, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, KmeansPPInitializer)

	var err error

	err = km.Push(data[0])
	err = km.Push(data[1])
	err = km.Push(data[2])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, err = km.Centroids()

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[3], false)

	if err == nil {
		t.Error("Expected workflow error")
	}

	km.Run(true)

	err = km.Push(data[3])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, _, err = km.Predict(data[4], true)

	if err != nil {
		t.Error("Expected no workflow error")
	}

	km.Close()

	err = km.Push(data[5])

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[5], true)

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[5], false)

	if err != nil {
		t.Error("Expected no workflow error")
	}
}

func TestKMeans_Conf(t *testing.T) {
	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	func() {
		defer testPanic()
		var conf = KMeansConf{Iter: -10, K: 3, Space: core.RealSpace{}}
		NewKMeans(conf, KmeansPPInitializer)
	}()

	func() {
		defer testPanic()
		var conf = KMeansConf{Iter: 10, K: -3, Space: core.RealSpace{}}
		NewKMeans(conf, KmeansPPInitializer)
	}()
}

func TestKMeans_Empty(t *testing.T) {
	var init = Clust{
		[]float64{0, 0, 0, 0, 0},
		[]float64{1000, 1000, 1000, 1000, 1000},
	}
	var conf = KMeansConf{Iter: 1, K: 2, Space: core.RealSpace{}}
	var km = NewKMeans(conf, init.Initializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(true)

	var clust, _ = km.Centroids()

	if !reflect.DeepEqual(clust[1], init[1]){
		t.Error("Expected mpty cluster")
	}
}
