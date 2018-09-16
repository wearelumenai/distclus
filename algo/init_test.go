package algo_test

import (
	"testing"
	"time"
	"golang.org/x/exp/rand"
	"distclus/real"
	"distclus/algo"
	"distclus/algo/zetest"
	"reflect"
)

func TestWeightedChoice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++ {
		sum[algo.WeightedChoice(w, src)] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000 {
		t.Errorf("Value out of range")
	}
}

func TestGivenInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = algo.GivenInitializer(4, zetest.TestPoints, real.RealSpace{}, src)

	if l := len(clust); l != 4 {
		t.Error("Expected 4 centers got", l)
	}

	for i := 0; i < 4; i++ {
		if !reflect.DeepEqual(clust[i], zetest.TestPoints[i]) {
			t.Error("Expected", zetest.TestPoints[i], "got", clust[i])
		}
	}
}

func TestKmeansPPInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = algo.KmeansPPInitializer(14, zetest.TestPoints, real.RealSpace{}, src)

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

func TestRandInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = algo.RandInitializer(14, zetest.TestPoints, real.RealSpace{}, src)

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
