package cosinus_test

import (
	"testing"

	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/cosinus"
)

func Test_ScalarProduct(t *testing.T) {
	var v1 = []float64{1., 1.}
	var v2 = []float64{1., 2.}
	var p = cosinus.ScalarProduct(v1, v2)
	if p != 3 {
		t.Error("result should be 3 got", p)
	}
}

func Test_Cosinus(t *testing.T) {
	var v1 = []float64{1., 1.}
	var v2 = []float64{1., 2.}
	var p = cosinus.Cosinus(v1, v2)
	if p < 0.948683 || p > 0.948684 {
		t.Error("result should be 0.9486832 got", p)
	}
}

func TestSpace_Dist(t *testing.T) {
	var space = cosinus.NewSpace()

	var v1 = []float64{1., 1.}
	var v2 = []float64{1., 2.}

	var d = space.Dist(v1, v2)
	if d < 0.051316 || d > 0.051317 {
		t.Error("result should be 0.9486832 got", d)
	}
}

func TestSpace_Dim(t *testing.T) {
	var space = cosinus.NewSpace()

	var v1 = []float64{1., 1.}

	var d = space.Dim([]core.Elemt{v1})

	if d != 2 {
		t.Error("dimension should be 2 got", d)
	}
}

func TestSpace_Combine(t *testing.T) {
	var space = cosinus.NewSpace()

	var v1 = []float64{1., 1.}
	var v2 = []float64{1., 2.}

	var v = space.Combine(v1, 4, v2, 1).([]float64)

	if v[0] != 1. || v[1] != 1.2 {
		t.Error("result should be [1., 1.2] got", v)
	}
}

func TestSpace_Copy(t *testing.T) {
	var space = cosinus.NewSpace()

	var v1 = []float64{1., 1.}
	var v2 = space.Copy(v1).([]float64)

	if v2[0] != 1. || v2[1] != 1. {
		t.Error("result should be [1., 1.] got", v2)
	}
}
