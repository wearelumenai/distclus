package series_test

import (
	"distclus/series"
	"distclus/vectors"
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
	var s11 = series.Resize(s1, 6, vectors.Space{})
	if !reflect.DeepEqual(s1, s11) {
		t.Error("resize error")
	}
	var s21 = series.Resize(s2, 7, vectors.Space{})
	if !reflect.DeepEqual(s2, s21) {
		t.Error("resize error")
	}
	var s12 = series.Resize(s1, 16, vectors.Space{})
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
