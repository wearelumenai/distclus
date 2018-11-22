package complex_test

import (
	"distclus/complex"
	"math/cmplx"
	"testing"
)

func TestComplexDist2And4(t *testing.T) {
	e1 := []complex128{2}
	e2 := []complex128{4}
	space := complex.Space{}
	val := space.Dist(e1, e2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestComplexDist0And0(t *testing.T) {
	e1 := []complex128{0}
	e2 := []complex128{0}
	space := complex.Space{}
	val := space.Dist(e1, e2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestComplexDist2_2And4_4(t *testing.T) {
	e1 := []complex128{2, 2}
	e2 := []complex128{4, 4}
	res := cmplx.Abs(cmplx.Sqrt(8))
	space := complex.Space{}
	val := space.Dist(e1, e2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestComplexCombine2x1And4x1(t *testing.T) {
	e1 := []complex128{2}
	e2 := []complex128{4}
	space := complex.Space{}
	space.Combine(e1, 1, e2, 1)
	if e1[0] != 3 {
		t.Errorf("Expected 3, got %v", e1)
	}
}

func TestComplexCombine2_1x2And4_2x2(t *testing.T) {
	e1 := []complex128{2, 1}
	e2 := []complex128{4, 2}
	space := complex.Space{}
	space.Combine(e1, 2, e2, 2)
	if e1[0] != 3 {
		t.Errorf("Expected 3, got %v", e1[0])
	}
	if e1[1] != 1.5 {
		t.Errorf("Expected 3/2, got %v", e1[1])
	}
}

func TestComplexCombine2_1x0And4_2x1(t *testing.T) {
	e1 := []complex128{2, 1}
	e2 := []complex128{4, 2}
	space := complex.Space{}
	space.Combine(e1, 0, e2, 1)
	if e1[0] != 4 {
		t.Errorf("Expected 3, got %v", e1[0])
	}
	if e1[1] != 2 {
		t.Errorf("Expected 3/2, got %v", e1[1])
	}
}

func TestComplexSpace_Copy(t *testing.T) {
	var e1 = []complex128{2, 1}
	sp := complex.Space{}
	var e2 = sp.Copy(e1).([]complex128)

	if e1[0] != e2[0] || e1[1] != e2[1] {
		t.Error("Expected same elements")
	}

	e2[0] = 3.
	e2[1] = 6.

	if e1[0] == e2[0] || e1[1] == e2[1] {
		t.Error("Expected different elements")
	}
}

func Test_NewSpace(t *testing.T) {
	space := complex.NewSpace(nil)

	if &space == nil {
		t.Error("no space created")
	}
}
