package kmeans_test

import (
	"reflect"
	"testing"
	"time"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/euclid"
	"github.com/wearelumenai/distclus/kmeans"

	"golang.org/x/exp/rand"
)

var TestPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

func TestInitializers(t *testing.T) {
	k := 1
	space := euclid.Space{}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))

	initializers := map[string]bool{
		"GiveN":   true,
		"given":   true,
		"pp":      true,
		"rand":    true,
		"unknown": false,
	}
	for name, test := range initializers {
		initializer := kmeans.CreateInitializer(name)
		if test {
			if initializer == nil {
				t.Error("miss an initializer")
			}
			centroids, err := initializer(k, TestPoints, space, src)
			if err != nil {
				t.Error("an error has been raised")
				return
			}
			if centroids == nil {
				t.Error("initializer does not return a centroid")
			}
		} else {
			if initializer != nil {
				t.Error("found an initializer")
			}
		}
	}
}

func TestWrongElementCount(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	_, err := kmeans.GivenInitializer(1, nil, euclid.Space{}, src)
	if err == nil {
		t.Error("Expected not check")
	}
}

func TestCheckK(t *testing.T) {
	// defer test.AssertPanic(t)
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	_, err := kmeans.GivenInitializer(0, TestPoints, euclid.Space{}, src)
	if err == nil {
		t.Error("initialization without errors")
	}
}

func TestWeightedChoice(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{10, 10, 80}
	var sum = make([]int, 3)
	var n = 100000
	for i := 0; i < n; i++ {
		var d, _ = kmeans.WeightedChoice(w, src)
		sum[d] += 1
	}
	var diff = sum[2] - 80000
	if diff < 79000 && diff > 81000 {
		t.Errorf("Value out of range")
	}
}

func TestWeightedChoiceErr(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var w = []float64{0, 0, 0}
	var _, err = kmeans.WeightedChoice(w, src)
	if err != kmeans.ErrNullSet {
		t.Error("Expected null set error")
	}
}

func TestGivenInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.GivenInitializer(4, TestPoints, euclid.Space{}, src)
	AssertCentroids(t, TestPoints[:4], clust)
}

func TestPPInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.PPInitializer(14, TestPoints, euclid.Space{}, src)
	AssertDistinctCentroids(t, clust)
}

func TestRandInitializer(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust, _ = kmeans.RandInitializer(14, TestPoints, euclid.Space{}, src)
	AssertDistinctCentroids(t, clust)
}

func TestPPIterErr(t *testing.T) {
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var _, err = kmeans.PPIter(TestPoints, TestPoints, euclid.Space{}, src)
	if err != kmeans.ErrNullSet {
		t.Error("Expected null set error")
	}
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
