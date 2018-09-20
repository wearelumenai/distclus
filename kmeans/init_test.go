package kmeans_test

import (
	"distclus/kmeans"
	"distclus/core"
	"distclus/real"
	"golang.org/x/exp/rand"
	"reflect"
	"testing"
	"time"
)

var TestPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

func TestWeightedChoice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++ {
		sum[kmeans.WeightedChoice(w, src)] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000 {
		t.Errorf("Value out of range")
	}
}

func TestGivenInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.GivenInitializer(4, TestPoints, real.RealSpace{}, src)
	AssertCentroids(t, TestPoints[:4], clust)
}

func TestKMeansPPInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.KMeansPPInitializer(14, TestPoints, real.RealSpace{}, src)
	AssertDistinctCentroids(t, clust)
}

func TestRandInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.RandInitializer(14, TestPoints, real.RealSpace{}, src)
	AssertDistinctCentroids(t, clust)
}

func AssertCentroids(t *testing.T, expected core.Clust, actual core.Clust) {
	if l := len(actual); l != 4 {
		t.Error("Expected 4 centers got", l)
	}
	for i := 0; i < 4; i++ {
		if !reflect.DeepEqual(actual[i], expected[i]) {
			t.Error("Expected", expected[i], "got", actual[i])
		}
	}
}

func AssertDistinctCentroids(t *testing.T, clust core.Clust) {
	if l := len(clust); l != 14 {
		t.Error("Expected 4 centers got", l)
	}
	for i := 0; i < 14; i++ {
		for j := 0; j < 14; j++ {
			if i != j && reflect.DeepEqual(clust[i], clust[j]) {
				t.Error("Expected distinct centers")
			}
		}
	}
}
