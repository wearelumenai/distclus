package core

import (
	"math"
	"testing"
)

func TestRealDist2And4(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := RealSpace{}
	val := space.Dist(e1, e2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestRealDist0And0(t *testing.T) {
	e1 := []float64{0}
	e2 := []float64{0}
	space := RealSpace{}
	val := space.Dist(e1, e2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestRealDist2_2And4_4(t *testing.T) {
	e1 := []float64{2, 2}
	e2 := []float64{4, 4}
	res := math.Sqrt(8)
	space := RealSpace{}
	val := space.Dist(e1, e2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestRealDist_And4_4(t *testing.T) {
	var e1 []float64
	e2 := []float64{4, 4}
	space := RealSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Dist(e1, e2)
}

func TestRealDist_And_(t *testing.T) {
	var e1 []float64
	var e2 []float64
	space := RealSpace{}
	var val float64
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Dist(e1, e2)
}

func TestRealDist2_1x2And4x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4}
	space := RealSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Dist(e1, e2)
}

func TestRealCombine2x1And4x1(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := RealSpace{}
	val := space.Combine(e1, 1, e2, 1).([]float64)
	if val[0] != 3 {
		t.Errorf("Expected 3, got %v", val)
	}
}

func TestRealCombine2_1x2And4_2x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := RealSpace{}
	val := space.Combine(e1, 2, e2, 2).([]float64)
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
	space := RealSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Combine(e1, 2, e2, 2).([]float64)
}

func TestRealCombine_And_(t *testing.T) {
	var e1 []float64
	var e2 []float64
	space := RealSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Combine(e1, 1, e2, 1)
}

func TestRealCombine2_1x0And4_2x1(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := RealSpace{}
	val := space.Combine(e1, 0, e2, 1).([]float64)
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
	space := RealSpace{}
	var val Elemt
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic, got %v", val)
		}
	}()
	val = space.Combine(e1, 0, e2, 0)
}