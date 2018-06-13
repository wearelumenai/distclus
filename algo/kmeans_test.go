package algo

import (
	"testing"
	"distclus/core"
	"golang.org/x/exp/rand"
)

func TestRandomInitKMeans(t *testing.T) {
	var data = make([]core.Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}
	var space = core.RealSpace{}
	var km = NewKMeans(3, 10, space, RandInitializer)
	for _, elt := range data {
		km.Push(elt)
	}
	km.Run()
	km.Close()
	var clust, _ = km.Centroids()
	if len(clust) != 3 {
		t.Errorf("Expected 3, got %v", 3)
	}
}

func TestDeterminedInitKMeans(t *testing.T) {
	var data = make([]core.Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}

	var localSpace = core.RealSpace{}

	var init = func(k int, elemts []core.Elemt, space core.Space, _ *rand.Rand) Clust {
		var clust = make(Clust, 3)
		clust[0] = []float64{7.2, 6, 8, 11, 10}
		clust[1] = []float64{-9, -10, -8, -8, -7.5}
		clust[2] = []float64{42, 41.2, 42.2, 40.2, 45}
		return clust
	}

	var km = NewKMeans(3, 10, localSpace, init)
	for _, elt := range data {
		km.Push(elt)
	}
	km.Run()
	km.Close()

	var clust, _ = km.Centroids()
	var assign = clust.Assign(data, localSpace)
	var c1 = len(assign[0])
	if c1 != 3 {
		t.Errorf("Expected 3, got %v", c1)
	}
	var c2 = len(assign[1])
	if c2 != 2 {
		t.Errorf("Expected 2, got %v", c2)
	}
	var c3 = len(assign[2])
	if c3 != 3 {
		t.Errorf("Expected 3, got %v", c3)
	}
}
