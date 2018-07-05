package par

import (
	"distclus/core"
	"testing"
	"reflect"
	"time"
	"golang.org/x/exp/rand"
	"distclus/algo"
	"math"
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

func TestKMeans_ParRun(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, algo.GivenInitializer, nil)

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

func TestKMeans_ParPredict(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 0, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, algo.GivenInitializer, nil)

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

func TestKMeans_ParPredict2(t *testing.T) {
	var seed = uint64(187236548914256543)
	rgen := rand.New(rand.NewSource(seed))
	var conf = algo.KMeansConf{Iter: 20, K: 3, Space: core.RealSpace{}, RGen: rgen}
	var km = NewKMeans(conf, algo.KmeansPPInitializer, nil)

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

		if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; i==0 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{-17. / 2, -20.5 / 2, -15. / 2, -16.5 / 2, -16.5 / 2}; i == 1 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; i == 2 && !almostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
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

func TestKMeans_ParAsync(t *testing.T) {
	var conf = algo.KMeansConf{Iter: 1 << 30, K: 3, Space: core.RealSpace{}}
	var km = NewKMeans(conf, algo.GivenInitializer, nil)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(true)

	time.Sleep(500 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, _, _ = km.Predict(obs, true)

	time.Sleep(700 * time.Millisecond)
	var cn, ixn, _ = km.Predict(obs, false)

	if reflect.DeepEqual(cn, c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = km.Predict(data[1], false)
	var _, ix5, _ = km.Predict(data[5], false)

	if ixn != ix5 || ixn != ix1 {
		t.Error("Expected same center")
	}

	var c0, _, _ = km.Predict(data[0], false)
	if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; !almostEqual(r, c0.([]float64)) {
		t.Error("Expected", r, "got", c0)
	}

	var c1, _, _ = km.Predict(data[1], false)
	if r := []float64{-26. / 3, -30.5 / 3, -23.3 / 3, -24.5 / 3, -24. / 3}; !almostEqual(r, c1.([]float64)) {
		t.Error("Expected", r, "got", c1)
	}

	var c2, _, _ = km.Predict(data[2], false)
	if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; !almostEqual(r, c2.([]float64)) {
		t.Error("Expected", r, "got", c2)
	}

	km.Close()
}
