package real_test

import (
	"distclus/real"
	"math"
	"testing"
)

func TestRealDist2And4(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := real.Space{}
	val := space.Dist(e1, e2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestRealDist0And0(t *testing.T) {
	e1 := []float64{0}
	e2 := []float64{0}
	space := real.Space{}
	val := space.Dist(e1, e2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestRealDist2_2And4_4(t *testing.T) {
	e1 := []float64{2, 2}
	e2 := []float64{4, 4}
	res := math.Sqrt(8)
	space := real.Space{}
	val := space.Dist(e1, e2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestRealCombine2x1And4x1(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := real.Space{}
	space.Combine(e1, 1, e2, 1)
	if e1[0] != 3 {
		t.Errorf("Expected 3, got %v", e1)
	}
}

func TestRealCombine2_1x2And4_2x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := real.Space{}
	space.Combine(e1, 2, e2, 2)
	if e1[0] != 3 {
		t.Errorf("Expected 3, got %v", e1[0])
	}
	if e1[1] != 1.5 {
		t.Errorf("Expected 3/2, got %v", e1[1])
	}
}

func TestRealCombine2_1x0And4_2x1(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := real.Space{}
	space.Combine(e1, 0, e2, 1)
	if e1[0] != 4 {
		t.Errorf("Expected 3, got %v", e1[0])
	}
	if e1[1] != 2 {
		t.Errorf("Expected 3/2, got %v", e1[1])
	}
}

func TestRealSpace_Copy(t *testing.T) {
	var e1 = []float64{2, 1}
	sp := real.Space{}
	var e2 = sp.Copy(e1).([]float64)

	if e1[0] != e2[0] || e1[1] != e2[1] {
		t.Error("Expected same elements")
	}

	e2[0] = 3.
	e2[1] = 6.

	if e1[0] == e2[0] || e1[1] == e2[1] {
		t.Error("Expected different elements")
	}
}
