package euclid_test

import (
	"math"
	"testing"

	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/euclid"
	"github.com/wearelumenai/distclus/v0/internal/test"
)

func TestVectorDist2And4(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := euclid.Space{}
	val := space.Dist(e1, e2)
	if val != 2 {
		t.Error("Expected 2, got ", val)
	}
}

func TestVectorDist0And0(t *testing.T) {
	e1 := []float64{0}
	e2 := []float64{0}
	space := euclid.Space{}
	val := space.Dist(e1, e2)
	if val != 0 {
		t.Error("Expected 0, got ", val)
	}
}

func TestVectorDist2_2And4_4(t *testing.T) {
	e1 := []float64{2, 2}
	e2 := []float64{4, 4}
	res := math.Sqrt(8)
	space := euclid.Space{}
	val := space.Dist(e1, e2)
	if val != res {
		t.Errorf("Expected %v, got %v", res, val)
	}
}

func TestVectorCombine2x1And4x1(t *testing.T) {
	e1 := []float64{2}
	e2 := []float64{4}
	space := euclid.Space{}
	var e3 = space.Combine(e1, 1, e2, 1).([]float64)
	if e3[0] != 3 {
		t.Errorf("Expected 3, got %v", e3)
	}
}

func TestVectorCombine2_1x2And4_2x2(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := euclid.Space{}
	var e3 = space.Combine(e1, 2, e2, 2).([]float64)
	if e3[0] != 3 {
		t.Errorf("Expected 3, got %v", e3[0])
	}
	if e3[1] != 1.5 {
		t.Errorf("Expected 3/2, got %v", e3[1])
	}
}

func TestVectorCombine2_1x0And4_2x1(t *testing.T) {
	e1 := []float64{2, 1}
	e2 := []float64{4, 2}
	space := euclid.Space{}
	var e3 = space.Combine(e1, 0, e2, 1).([]float64)
	if e3[0] != 4 {
		t.Errorf("Expected 3, got %v", e3[0])
	}
	if e3[1] != 2 {
		t.Errorf("Expected 3/2, got %v", e3[1])
	}
}

func TestVectorSpace_Copy(t *testing.T) {
	var e1 = []float64{2, 1}
	sp := euclid.Space{}
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

func Test_NewSpace(t *testing.T) {
	space := euclid.NewSpace()

	if &space == nil {
		t.Error("no space created")
	}
}

func Test_Dim(t *testing.T) {
	space := euclid.NewSpace()

	data := make([]core.Elemt, 1)
	data[0] = []float64{}

	dim := space.Dim(data)

	test.AssertEqual(t, dim, 0)

	data[0] = []float64{1., 2., 3.}

	dim = space.Dim(data)

	test.AssertEqual(t, dim, 3)
}
