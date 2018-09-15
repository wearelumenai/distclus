package algo

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"reflect"
	"time"
	"golang.org/x/exp/rand"
	"math"
)

func TestKMeans_Centroids(t *testing.T) {
	var conf = KMeansConf{Iter: 0, K: 4, Space: real.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < conf.K; i++ {
		if !reflect.DeepEqual(clust[i], testVectors[i]) {
			t.Error("Expected", testVectors[i], "got", clust[i])
		}
	}

	km.Close()
}

func TestKMeans_Run(t *testing.T) {
	var conf = KMeansConf{Iter: 1, K: 3, Space: real.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < conf.K; i++ {
		if reflect.DeepEqual(clust[i], testVectors[i]) {
			t.Error("Expected average got", clust[i])
		}
	}

	km.Close()
}

func TestKMeans_Predict(t *testing.T) {
	var conf = KMeansConf{Iter: 0, K: 3, Space: real.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	km.Run(false)

	for i, elemt := range testVectors {
		var c, idx, _ = km.Predict(elemt, false)

		if i == 0 || i == 3 || i == 4 {
			if idx != 0 || !reflect.DeepEqual(c, testVectors[0]) {
				t.Error("Expected center 0")
			}
		}

		if i == 1 || i == 5 {
			if idx != 1 || !reflect.DeepEqual(c, testVectors[1]) {
				t.Error("Expected center 1")
			}
		}

		if i == 2 || i == 6 || i == 7 {
			if idx != 2 || !reflect.DeepEqual(c, testVectors[2]) {
				t.Error("Expected center 2")
			}
		}
	}

	km.Close()
}

func almostEqual(e1 []float64, e2 []float64) bool {
	if len(e1) != len(e2) {
		return false
	}
	for i := 0; i < len(e1); i++ {
		if math.Abs(e1[i]-e2[i]) > 1e-6 {
			return false
		}
	}
	return true
}

func TestKMeans_Predict2(t *testing.T) {
	var seed = uint64(187236548914256543)
	rgen := rand.New(rand.NewSource(seed))
	var conf = KMeansConf{Iter: 20, K: 3, Space: real.RealSpace{}, RGen: rgen}
	var km = NewKMeans(conf, KmeansPPInitializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = km.Predict(testVectors[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}

		if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; i == 0 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{-17. / 2, -20.5 / 2, -15. / 2, -16.5 / 2, -16.5 / 2}; i == 1 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; i == 2 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}
	}

	for i := 3; i < len(testVectors); i++ {
		var c, idx, _ = km.Predict(testVectors[i], false)

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
	var conf = KMeansConf{Iter: 1 << 30, K: 4, Space: real.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	if km.status != core.Created {
		t.Error("Expected status", core.Created, "got", km.status)
	}

	km.Run(true)
	time.Sleep(300*time.Millisecond)

	if km.status != core.Running {
		t.Error("Expected status", core.Running, "got", km.status)
	}

	km.Close()

	if km.status != core.Closed {
		t.Error("Expected status", core.Closed, "got", km.status)
	}
}

func TestKMeans_Async(t *testing.T) {
	var conf = KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = NewKMeans(conf, GivenInitializer, nil)

	km.Run(true)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	time.Sleep(500 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, _, _ = km.Predict(obs, true)

	time.Sleep(700 * time.Millisecond)
	var cn, ixn, _ = km.Predict(obs, false)

	if reflect.DeepEqual(cn, c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = km.Predict(testVectors[1], false)
	var _, ix5, _ = km.Predict(testVectors[5], false)

	if ixn != ix5 || ixn != ix1 {
		t.Error("Expected same center")
	}

	var c0, _, _ = km.Predict(testVectors[0], false)
	if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; !almostEqual(r, c0.([]float64)) {
		t.Error("Expected", r, "got", c0)
	}

	var c1, _, _ = km.Predict(testVectors[1], false)
	if r := []float64{-26. / 3, -30.5 / 3, -23.3 / 3, -24.5 / 3, -24. / 3}; !almostEqual(r, c1.([]float64)) {
		t.Error("Expected", r, "got", c1)
	}

	var c2, _, _ = km.Predict(testVectors[2], false)
	if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; !almostEqual(r, c2.([]float64)) {
		t.Error("Expected", r, "got", c2)
	}

	km.Close()
}

func TestKMeans_Workflow(t *testing.T) {
	var conf = KMeansConf{Iter: 1 << 30, K: 3, Space: real.RealSpace{}}
	var km = NewKMeans(conf, KmeansPPInitializer, nil)

	var err error

	err = km.Push(testVectors[0])
	err = km.Push(testVectors[1])
	err = km.Push(testVectors[2])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, err = km.Centroids()

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(testVectors[3], false)

	if err == nil {
		t.Error("Expected workflow error")
	}

	km.Run(true)
	time.Sleep(300*time.Millisecond)

	err = km.Push(testVectors[3])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, _, err = km.Predict(testVectors[4], true)

	if err != nil {
		t.Error("Expected no workflow error")
	}

	km.Close()

	err = km.Push(testVectors[5])

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(testVectors[5], true)

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(testVectors[5], false)

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
		var conf = KMeansConf{Iter: -10, K: 3, Space: real.RealSpace{}}
		var km = NewKMeans(conf, KmeansPPInitializer, nil)
		km.Run(false)
	}()

	func() {
		defer testPanic()
		var conf = KMeansConf{Iter: 10, K: -3, Space: real.RealSpace{}}
		NewKMeans(conf, KmeansPPInitializer, nil)
	}()
}

func TestKMeans_Empty(t *testing.T) {
	var init = core.Clust{
		[]float64{0, 0, 0, 0, 0},
		[]float64{1000, 1000, 1000, 1000, 1000},
	}
	var conf = KMeansConf{Iter: 1, K: 2, Space: real.RealSpace{}}
	var km = NewKMeans(conf, init.Initializer, nil)

	for _, elemt := range testVectors {
		km.Push(elemt)
	}

	km.Run(true)
	time.Sleep(300*time.Millisecond)

	var clust, _ = km.Centroids()

	if !reflect.DeepEqual(clust[1], init[1]) {
		t.Error("Expected empty cluster")
	}
}
