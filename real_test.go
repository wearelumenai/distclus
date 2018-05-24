package clustering_go

import (
	"math"
	"testing"
)

func TestRealDist2And4(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := realSpace{}
	val := space.dist(e1, e2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestRealDist0And0(t *testing.T) {
	e1 := []float64{0}
	e2 := []float64{0}
	space := realSpace{}
	val := space.dist(e1, e2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestRealDist2_2And4_4(t *testing.T) {
	e1 := []float64{2, 2}
	e2 := []float64{4, 4}
	res := math.Sqrt(8)
	space := realSpace{}
	val := space.dist(e1, e2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestRealDist_And4_4(t *testing.T) {
	var e1 []float64
	e2 := []float64{4, 4}
	space := realSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(e1, e2)
}

func TestRealDist_And_(t *testing.T) {
	var e1 []float64
	var e2 []float64
	space := realSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(e1, e2)
}

func TestRealDist2_1x2And4x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4}
	space := realSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(e1, e2)
}

func TestRealCombine2x1And4x1(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := realSpace{}
	val := space.combine(e1, 1, e2, 1).([]float64)
	if val[0] != 3 {
		t.Errorf("Expected 3, got %v", val)
	}
}

func TestRealCombine2_1x2And4_2x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := realSpace{}
	val := space.combine(e1, 2, e2, 2).([]float64)
	if val[0] != 3 {
		t.Errorf("Expected 3, got %v", val[0])
	}
	if val[1] != 1.5 {
		t.Errorf("Expected 3/2, got %v", val[1])
	}
}

func TestRealCombine2_1x2And4x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4}
	space := realSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(e1, 2, e2, 2).([]float64)
}

func TestRealCombine_And_(t *testing.T) {
	var e1 []float64
	var e2 []float64
	space := realSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(e1, 1, e2, 1)
}

func TestRealCombine2_1x0And4_2x1(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := realSpace{}
	val := space.combine(e1, 0, e2, 1).([]float64)
	if val[0] != 4 {
		t.Errorf("Expected 3, got %v", val[0])
	}
	if val[1] != 2 {
		t.Errorf("Expected 3/2, got %v", val[1])
	}
}

func TestRealCombine2_1x0And4_2x0(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := realSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(e1, 0, e2, 0)
}

func TestRandomInitKMeans(t *testing.T) {
	var data = make([]Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}
	var space = realSpace{}
	var km = NewKMeans(3, 10, space, randomInit)
	for _, elt := range data {
		km.Push(elt)
	}
	km.Run()
	km.Close()
	var clusters = km.clustering.clust
	if len(clusters) != 3 {
		t.Errorf("Expected 3, got %v", 3)
	}
}

func TestDeterminedInitKMeans(t *testing.T) {
	var data = make([]Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}
	var localSpace = realSpace{}
	var init = func(k int, elemts []Elemt, space space) clustering {
		var centroids = make([]Elemt, 3)
		var clusters = make([][]Elemt, 3)
		centroids[0] = []float64{7.2, 6, 8, 11, 10}
		centroids[1] = []float64{-9, -10, -8, -8, -7.5}
		centroids[2] = []float64{42, 41.2, 42.2, 40.2, 45}
		for _, elemt := range elemts {
			var idx = assign(elemt, centroids, space)
			clusters[idx] = append(clusters[idx], elemt)
		}
		var c, _ = newClustering(centroids, clusters)
		return c
	}
	var km = NewKMeans(3, 10, localSpace, init)
	for _, elt := range data {
		km.Push(elt)
	}
	km.Run()
	km.Close()
	var clusters = km.clustering.clust
	var c1 = len(clusters[0].elemts)
	if c1 != 3 {
		t.Errorf("Expected 3, got %v", c1)
	}
	var c2 = len(clusters[1].elemts)
	if c2 != 2 {
		t.Errorf("Expected 2, got %v", c2)
	}
	var c3 = len(clusters[2].elemts)
	if c3 != 3 {
		t.Errorf("Expected 3, got %v", c3)
	}
}

func TestProposalLossNorm2(t *testing.T) {
	var localSpace = realSpace{}
	var centroids = make([]Elemt, 2)
	centroids[0] = []float64{1}
	centroids[1] = []float64{42}
	var clusters = make([][]Elemt, 2)
	clusters[0] = make([]Elemt, 6)
	clusters[1] = make([]Elemt, 4)
	clusters[0][0] = []float64{2}
	clusters[0][1] = []float64{1}
	clusters[0][2] = []float64{3.1}
	clusters[0][3] = []float64{-1}
	clusters[0][4] = []float64{2.4}
	clusters[0][5] = []float64{1.4}
	clusters[1][0] = []float64{41}
	clusters[1][1] = []float64{42.3}
	clusters[1][2] = []float64{43}
	clusters[1][3] = []float64{42.9}
	var clustering, _ = newClustering(centroids, clusters)
	var proposal = proposal{k: 2, clustering: clustering}
	var loss = proposal.loss(1, localSpace)
	var res float64
	for i := range clusters {
		for _, value := range clusters[i] {
			var x = value.([]float64)[0]
			var y = centroids[i].([]float64)[0]
			res += math.Sqrt(math.Pow(x-y, 2))
		}
	}
	res /= 10
	if loss != res {
		t.Errorf("Expected %v, got %v", res, loss)
	}
}
