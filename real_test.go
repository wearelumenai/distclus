package clustering_go

import (
	"testing"
	"math"
)

func TestRealDist2And4(t *testing.T) {
	n1 := []float64{2}
	n2 := []float64{4}
	space := realSpace{}
	val := space.dist(n1, n2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestRealDist0And0(t *testing.T) {
	n1 := []float64{0}
	n2 := []float64{0}
	space := realSpace{}
	val := space.dist(n1, n2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestRealDist2_2And4_4(t *testing.T) {
	n1 := []float64{2, 2}
	n2 := []float64{4, 4}
	res := math.Sqrt(8)
	space := realSpace{}
	val := space.dist(n1, n2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestRealDist_And4_4(t *testing.T) {
	var n1 []float64
	n2 := []float64{4, 4}
	space := realSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(n1, n2)
}

func TestRealDist_And_(t *testing.T) {
	var n1 []float64
	var n2 []float64
	space := realSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(n1, n2)
}

func TestRealDist2_1x2And4x2(t *testing.T) {
	n1 := []float64{2, 1}
	n2 := []float64{4}
	space := realSpace{}
	var val node
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.dist(n1, n2)
}

func TestRealCombine2x1And4x1(t *testing.T) {
	n1 := []float64{2}
	n2 := []float64{4}
	space := realSpace{}
	val := space.combine(n1, 1, n2, 1).([]float64)
	if val[0] != 3 {
		t.Errorf("Expected 3, got %v", val)
	}
}

func TestRealCombine2_1x2And4_2x2(t *testing.T) {
	n1 := []float64{2, 1}
	n2 := []float64{4, 2}
	space := realSpace{}
	val := space.combine(n1, 2, n2, 2).([]float64)
	if val[0] != 3 {
		t.Errorf("Expected 3, got %v", val[0])
	}
	if val[1] != 1.5 {
		t.Errorf("Expected 3/2, got %v", val[1])
	}
}

func TestRealCombine2_1x2And4x2(t *testing.T) {
	n1 := []float64{2, 1}
	n2 := []float64{4}
	space := realSpace{}
	var val node
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(n1, 2, n2, 2).([]float64)
}

func TestRealCombine_And_(t *testing.T) {
	var n1 []float64
	var n2 []float64
	space := realSpace{}
	var val node
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(n1, 1, n2, 1)
}

func TestRealCombine2_1x0And4_2x1(t *testing.T) {
	n1 := []float64{2, 1}
	n2 := []float64{4, 2}
	space := realSpace{}
	val := space.combine(n1, 0, n2, 1).([]float64)
	if val[0] != 4 {
		t.Errorf("Expected 3, got %v", val[0])
	}
	if val[1] != 2 {
		t.Errorf("Expected 3/2, got %v", val[1])
	}
}

func TestRealCombine2_1x0And4_2x0(t *testing.T) {
	n1 := []float64{2, 1}
	n2 := []float64{4, 2}
	space := realSpace{}
	var val node
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.combine(n1, 0, n2, 0)
}

func TestRadomInitKMeans(t *testing.T) {
	var nodes = make([]node, 8)
	nodes[0] = []float64{7.2, 6, 8, 11, 10}
	nodes[1] = []float64{9, 8, 7, 7.5, 10}
	nodes[2] = []float64{7.2, 6, 8, 11, 10}
	nodes[3] = []float64{-9, -10, -8, -8, -7.5}
	nodes[4] = []float64{-8, -10.5, -7, -8.5, -9}
	nodes[5] = []float64{42, 41.2, 42, 40.2, 45}
	nodes[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	nodes[7] = []float64{50, 51.2, 49, 40, 45.2}
	space := realSpace{}
	clusters := kMeans(nodes, space, 3, 10, randomInit)
	if len(clusters) != 3 {
		t.Errorf("Expected 3, got %v", 3)
	}
}

func TestDeterminedInitKMeans(t *testing.T) {
	var nodes = make([]node, 8)
	nodes[0] = []float64{7.2, 6, 8, 11, 10}
	nodes[1] = []float64{9, 8, 7, 7.5, 10}
	nodes[2] = []float64{7.2, 6, 8, 11, 10}
	nodes[3] = []float64{-9, -10, -8, -8, -7.5}
	nodes[4] = []float64{-8, -10.5, -7, -8.5, -9}
	nodes[5] = []float64{42, 41.2, 42, 40.2, 45}
	nodes[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	nodes[7] = []float64{50, 51.2, 49, 40, 45.2}
	space := realSpace{}
	init := func(n int, nodes []node) []node{
		var c = make([]node, 3)
		c[0] = []float64{7.2, 6, 8, 11, 10}
		c[1] = []float64{-9, -10, -8, -8, -7.5}
		c[2] = []float64{42, 41.2, 42.2, 40.2, 45}
		return c
	}
	clusters := kMeans(nodes, space, 3, 10, init)
	if len(clusters[0]) != 3 {
		t.Errorf("Expected 3, got %v", len(clusters[0]))
	}
	if len(clusters[1]) != 2 {
		t.Errorf("Expected 2, got %v", len(clusters[1]))
	}
	if len(clusters[2]) != 3 {
		t.Errorf("Expected 3, got %v", len(clusters[2]))
	}
}

func TestBadInitKMeans(t *testing.T) {
	var nodes = make([]node, 8)
	nodes[0] = []float64{7.2, 6, 8, 11, 10}
	nodes[1] = []float64{9, 8, 7, 7.5, 10}
	nodes[2] = []float64{7.2, 6, 8, 11, 10}
	nodes[3] = []float64{-9, -10, -8, -8, -7.5}
	nodes[4] = []float64{-8, -10.5, -7, -8.5, -9}
	nodes[5] = []float64{42, 41.2, 42, 40.2, 45}
	nodes[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	nodes[7] = []float64{50, 51.2, 49, 40, 45.2}
	space := realSpace{}
	init := func(n int, nodes []node) []node{
		return make([]node, 0)
	}
	var clusters map[int][]node
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", clusters)
		}
	}()
	clusters = kMeans(nodes, space, 3, 10, init)
}