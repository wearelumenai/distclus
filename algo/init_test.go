package algo

import (
	"testing"
	"time"
	"golang.org/x/exp/rand"
	"distclus/core"
	"distclus/real"
	"reflect"
)

var testPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

var testVectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

func TestWeightedChoice(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++ {
		sum[WeightedChoice(w, src)] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000 {
		t.Errorf("Value out of range")
	}
}

func TestGivenInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = GivenInitializer(4, testPoints, real.RealSpace{}, src)

	if l := len(clust); l != 4 {
		t.Error("Expected 4 centers got", l)
	}

	for i := 0; i < 4; i++ {
		if !reflect.DeepEqual(clust[i], testPoints[i]) {
			t.Error("Expected", testPoints[i], "got", clust[i])
		}
	}
}

func TestKmeansPPInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = KmeansPPInitializer(14, testPoints, real.RealSpace{}, src)

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
	var clust, _ = RandInitializer(14, testPoints, real.RealSpace{}, src)

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

func TestCheckInitialization(t *testing.T) {

	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	if check(15, testVectors) {
		t.Error("Expected error")
	}

	func() {
		defer testPanic()

		check(0, testVectors)
	}()
}
