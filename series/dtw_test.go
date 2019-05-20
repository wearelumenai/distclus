package series_test

import (
	"distclus/series"
	"distclus/vectors"
	"math"
	"reflect"
	"testing"
)

var space = vectors.Space{}
var s1 = [][]float64{{1}, {1}, {2}, {3}, {2}, {0}}
var s2 = [][]float64{{0}, {1}, {1}, {2}, {3}, {4}, {2}}
var cumCost = [][]float64{
	{1, 1, 1, 2, 4, 7, 8},
	{2, 1, 1, 2, 4, 7, 8},
	{4, 2, 2, 1, 2, 4, 4},
	{7, 4, 4, 2, 1, 2, 3},
	{9, 5, 5, 2, 2, 3, 2},
	{9, 6, 6, 4, 5, 6, 4},
}
var path = [][]int{
	{5, 6},
	{4, 6},
	{3, 5},
	{3, 4},
	{2, 3},
	{1, 2},
	{0, 1},
	{0, 0},
}
var dba = [][]float64{{0.5}, {1}, {1.5}, {2.5}, {3.5}, {2}}

var inf = math.Inf(1)
var cumCost1 = [][]float64{
	{1, 1, inf, inf, inf, inf, inf},
	{2, 1, 1, inf, inf, inf, inf},
	{inf, 2, 2, 1, inf, inf, inf},
	{inf, inf, 4, 2, 1, inf, inf},
	{inf, inf, inf, 2, 2, 3, inf},
	{inf, inf, inf, inf, 5, 6, 5},
}
var path1 = [][]int{
	{5, 6},
	{4, 5},
	{3, 4},
	{2, 3},
	{1, 2},
	{0, 1},
	{0, 0},
}
var dba1 = [][]float64{{0.5}, {1}, {1.5}, {2.5}, {3}, {2}}

func Test_DTWMatrix1(t *testing.T) {
	var dtw = series.NewDTW(s1, s2, space)
	for i1 := range s1 {
		for i2 := range s2 {
			if dtw.CumCost(i1, i2) != cumCost[i1][i2] {
				t.Error("distance error for", i1, i2)
			}
		}
	}
}

func Test_DTWMatrix2(t *testing.T) {
	var dtw = series.NewDTW(s2, s1, space)
	for i2 := range s2 {
		for i1 := range s1 {
			if dtw.CumCost(i2, i1) != cumCost[i1][i2] {
				t.Error("distance error for", i2, i1)
			}
		}
	}
}

func Test_DTWMatrixWindow1(t *testing.T) {
	var dtw = series.NewDTWWindow(s1, s2, space, 1)
	for i1 := range s1 {
		for i2 := range s2 {
			if dtw.CumCost(i1, i2) != cumCost1[i1][i2] {
				t.Error("distance error for", i1, i2)
				return
			}
		}
	}
}

func Test_DTWMatrixWindow2(t *testing.T) {
	var dtw = series.NewDTWWindow(s2, s1, space, 1)
	for i2 := range s2 {
		for i1 := range s1 {
			if dtw.CumCost(i2, i1) != cumCost1[i1][i2] {
				t.Error("distance error for", i2, i1)
				return
			}
		}
	}
}

func Test_DTWDist1(t *testing.T) {
	var dtw = series.NewDTW(s1, s2, space)
	if dtw.Dist() != cumCost[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDistWindow1(t *testing.T) {
	var dtw = series.NewDTWWindow(s1, s2, space, 1)
	if dtw.Dist() != cumCost1[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDist2(t *testing.T) {
	var dtw = series.NewDTW(s2, s1, space)
	if dtw.Dist() != cumCost[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDistWindow2(t *testing.T) {
	var dtw = series.NewDTWWindow(s2, s1, space, 1)
	if dtw.Dist() != cumCost1[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWPath1(t *testing.T) {
	var dtw = series.NewDTW(s1, s2, space)
	for i, p := range dtw.Path() {
		if p[0] != path[i][0] || p[1] != path[i][1] {
			t.Error("path error")
		}
	}
}

func Test_DTWPathWindow1(t *testing.T) {
	var dtw = series.NewDTWWindow(s1, s2, space, 1)
	for i, p := range dtw.Path() {
		if p[0] != path1[i][0] || p[1] != path1[i][1] {
			t.Error("path error")
		}
	}
}

func Test_DTWPath2(t *testing.T) {
	var dtw = series.NewDTW(s2, s1, space)
	for i, p := range dtw.Path() {
		if p[0] != path[i][1] || p[1] != path[i][0] {
			t.Error("path error")
		}
	}
}

func Test_DTWPathWindow2(t *testing.T) {
	var dtw = series.NewDTWWindow(s2, s1, space, 1)
	for i, p := range dtw.Path() {
		if p[0] != path1[i][1] || p[1] != path1[i][0] {
			t.Error("path error")
		}
	}
}

func Test_DTWDBA1(t *testing.T) {
	var dtw = series.NewDTW(s1, s2, space)
	if !reflect.DeepEqual(dba, dtw.DBA(1, 1)) {
		t.Error("dba error")
	}
}

func Test_DTWDBAWindow1(t *testing.T) {
	var dtw = series.NewDTWWindow(s1, s2, space, 1)
	if !reflect.DeepEqual(dba1, dtw.DBA(1, 1)) {
		t.Error("dba error")
	}
}

func Test_DTWDBA2(t *testing.T) {
	var dtw = series.NewDTW(s2, s1, space)
	if !reflect.DeepEqual(dba, dtw.DBA(1, 1)) {
		t.Error("dba error")
	}
}

func Test_DTWDBAWindow2(t *testing.T) {
	var dtw = series.NewDTWWindow(s2, s1, space, 1)
	if !reflect.DeepEqual(dba1, dtw.DBA(1, 1)) {
		t.Error("dba error")
	}
}

func Test_DTWResize(t *testing.T) {
	var s12 = series.Resize(s1, 16, space)
	var dtw = series.NewDTW(s12, s2, space)
	d := dtw.DBA(1, 1)
	if len(d) != 11 {
		t.Error("dba error")
	}
}
