package series_test

import (
	"distclus/series"
	"reflect"
	"testing"
)

var se = [][]float64{
	{1.}, {1.}, {1.},
	{1.}, {4. / 3.}, {5. / 3.},
	{2.}, {7. / 3.}, {8. / 3.},
	{3.}, {8. / 3.}, {7. / 3.},
	{2.}, {4. / 3.}, {2. / 3.}, {0.}}

var si = [][]float64{{0.5}, {1}, {1}, {2}, {3}, {3.5}, {2}, {1}}
var ix = []int{0, 1, 3, 5, 7, 8, 10, 11}
var iw = 2

func Test_Resize(t *testing.T) {
	var s11 = series.Resize(s1, 6, space)
	if !reflect.DeepEqual(s1, s11) {
		t.Error("resize error")
	}
	var s21 = series.Resize(s2, 7, space)
	if !reflect.DeepEqual(s2, s21) {
		t.Error("resize error")
	}
	var s12 = series.Resize(s1, 16, space)
	if !reflect.DeepEqual(se, s12) {
		t.Error("resize error")
	}
}

func Test_Interpolate(t *testing.T) {
	var si1 = series.Interpolate(si, ix, iw, space)
	if !reflect.DeepEqual(dba, si1) {
		t.Error("interpolate error")
	}
}

func Test_ShrinkLongest(t *testing.T) {
	var s12 = series.Resize(s1, 16, space)
	var s211, s121 = series.ShrinkLongest(s2, s12, space, 3)
	if len(s211) != 7 {
		t.Error("shrink error")
	}
	if len(s121) != 10 {
		t.Error("shrink error")
	}
	var s122, s212 = series.ShrinkLongest(s12, s2, space, 3)
	if len(s212) != 7 {
		t.Error("shrink error")
	}
	if len(s122) != 10 {
		t.Error("shrink error")
	}
}
