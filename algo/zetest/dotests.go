package zetest

import (
	"distclus/core"
	"distclus/algo"
	"testing"
	"time"
	"reflect"
	"math"
)

var TestPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

var TestVectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

func DoTest_Centroids(t *testing.T, algo core.OnlineClust) {
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	algo.Run(false)
	var clust, _ = algo.Centroids()

	assertCentroids(t, TestVectors[:len(clust)], clust)

	algo.Close()
}

func assertCentroids(t *testing.T, expected core.Clust, actual core.Clust) {

	if len(actual) != len(expected) {
		t.Error("Expected ", len(expected), "centroids got", len(actual))
		return
	}

	for i := 0; i < len(actual); i++ {
		if !reflect.DeepEqual(actual[i], expected[i]) {
			t.Error("Expected", expected[i], "got", actual[i])
		}
	}
}

func DoTest_Run_Sync(t *testing.T, algo core.OnlineClust) {
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	algo.Run(false)
	var clust, _ = algo.Centroids()

	for i := 0; i < len(clust); i++ {
		if reflect.DeepEqual(clust[i], TestVectors[i]) {
			t.Error("Expected average got", clust[i])
		}
	}

	algo.Close()
}

func DoTest_Predict_Given(t *testing.T, builder func(core.Initializer) core.OnlineClust) core.OnlineClust {
	var algo = builder(algo.GivenInitializer)
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	algo.Run(false)

	for i, elemt := range TestVectors {
		var c, idx, _ = algo.Predict(elemt, false)

		if i == 0 || i == 3 || i == 4 {
			if idx != 0 || !reflect.DeepEqual(c, TestVectors[0]) {
				t.Error("Expected center 0")
			}
		}

		if i == 1 || i == 5 {
			if idx != 1 || !reflect.DeepEqual(c, TestVectors[1]) {
				t.Error("Expected center 1")
			}
		}

		if i == 2 || i == 6 || i == 7 {
			if idx != 2 || !reflect.DeepEqual(c, TestVectors[2]) {
				t.Error("Expected center 2")
			}
		}
	}

	algo.Close()
	return algo
}

func DoTest_Predict_KMeansPP(t *testing.T, builder func(core.Initializer) core.OnlineClust) core.OnlineClust {
	var algo = builder(algo.KmeansPPInitializer)
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	algo.Run(false)
	var clust, _ = algo.Centroids()
	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = algo.Predict(TestVectors[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}
	}

	for i := 3; i < len(TestVectors); i++ {
		var c, idx, _ = algo.Predict(TestVectors[i], false)

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

	algo.Close()
	return algo
}

func DoTest_Predict_Centroids(t *testing.T, km *algo.KMeans) {
	var clust, _ = km.Centroids()

	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = km.Predict(TestVectors[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}

		if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; i == 0 && !AlmostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{-17. / 2, -20.5 / 2, -15. / 2, -16.5 / 2, -16.5 / 2}; i == 1 && !AlmostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}

		if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; i == 2 && !AlmostEqual(r, c.([]float64)) {
			t.Error("Expected", r, "got", c)
		}
	}
}

func DoTest_Run_Async(t *testing.T, algo core.OnlineClust) {
	algo.Run(true)

	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	time.Sleep(700 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, _, _ = algo.Predict(obs, true)

	time.Sleep(1000 * time.Millisecond)
	var cn, ixn, _ = algo.Predict(obs, false)

	if reflect.DeepEqual(cn, c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = algo.Predict(TestVectors[1], false)
	var _, ix5, _ = algo.Predict(TestVectors[5], false)

	if ixn != ix5 || ixn != ix1 {
		t.Error("Expected same center")
	}

	algo.Close()
}

func DoTest_Run_Async_Centroids(t *testing.T, km *algo.KMeans) {
	var c0, _, _ = km.Predict(TestVectors[0], false)
	if r := []float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3}; !AlmostEqual(r, c0.([]float64)) {
		t.Error("Expected", r, "got", c0)
	}

	var c1, _, _ = km.Predict(TestVectors[1], false)
	if r := []float64{-26. / 3, -30.5 / 3, -23.3 / 3, -24.5 / 3, -24. / 3}; !AlmostEqual(r, c1.([]float64)) {
		t.Error("Expected", r, "got", c1)
	}

	var c2, _, _ = km.Predict(TestVectors[2], false)
	if r := []float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3}; !AlmostEqual(r, c2.([]float64)) {
		t.Error("Expected", r, "got", c2)
	}
}

func DoTest_Workflow(t *testing.T, algo core.OnlineClust) {
	var err error

	err = algo.Push(TestVectors[0])
	err = algo.Push(TestVectors[1])
	err = algo.Push(TestVectors[2])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, err = algo.Centroids()

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = algo.Predict(TestVectors[3], false)

	if err == nil {
		t.Error("Expected workflow error")
	}

	algo.Run(true)
	time.Sleep(300*time.Millisecond)

	err = algo.Push(TestVectors[3])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, _, err = algo.Predict(TestVectors[4], true)

	if err != nil {
		t.Error("Expected no workflow error")
	}

	algo.Close()

	err = algo.Push(TestVectors[5])

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = algo.Predict(TestVectors[5], true)

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = algo.Predict(TestVectors[5], false)

	if err != nil {
		t.Error("Expected no workflow error")
	}
}

func DoTest_Empty(t *testing.T, builder func(core.Initializer) core.OnlineClust) {
	var init = core.Clust{
		[]float64{0, 0, 0, 0, 0},
		[]float64{1000, 1000, 1000, 1000, 1000},
	}
	var algo = builder(init.Initializer)

	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}

	algo.Run(true)
	time.Sleep(300*time.Millisecond)

	var clust, _ = algo.Centroids()

	if !reflect.DeepEqual(clust[1], init[1]) {
		t.Error("Expected empty cluster")
	}
}

func AlmostEqual(e1 []float64, e2 []float64) bool {
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

func TestPanic(t *testing.T) {
	if x := recover(); x == nil {
		t.Error("Expected error")
	}
}
