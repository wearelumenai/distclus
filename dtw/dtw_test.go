package dtw_test

import (
	"math"
	"testing"

	"go.lumenai.fr/distclus/v0/dtw"
)

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
var dbaw = [][]float64{{1. / 3}, {1.}, {4. / 3}, {7. / 3}, {10. / 3}, {23. / 9}}

var inf = math.Inf(1)
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
var dbaw1 = [][]float64{{2. / 3}, {1.}, {5. / 3}, {8. / 3}, {25. / 9}, {4. / 3}}

func Test_DTWDist1(t *testing.T) {
	var dist = dtw.NewDTW(s1, s2, space)
	if dist.Dist() != cumCost[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDistWindow1(t *testing.T) {
	var dist = dtw.NewDTWWindow(s1, s2, space, 1)
	if dist.Dist() != cumCost1[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDist2(t *testing.T) {
	var dist = dtw.NewDTW(s2, s1, space)
	if dist.Dist() != cumCost[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWDistWindow2(t *testing.T) {
	var dist = dtw.NewDTWWindow(s2, s1, space, 1)
	if dist.Dist() != cumCost1[5][6] {
		t.Error("distance error")
	}
}

func Test_DTWPath1(t *testing.T) {
	var dist = dtw.NewDTW(s1, s2, space)
	for i, p := range dist.Path() {
		if p.End0 != path[i][0] || p.End1 != path[i][1] {
			t.Error("path error")
		}
	}
}

func Test_DTWPathWindow1(t *testing.T) {
	var dist = dtw.NewDTWWindow(s1, s2, space, 1)
	for i, p := range dist.Path() {
		if p.End0 != path1[i][0] || p.End1 != path1[i][1] {
			t.Error("path error")
		}
	}
}

func Test_DTWPath2(t *testing.T) {
	var dist = dtw.NewDTW(s2, s1, space)
	for i, p := range dist.Path() {
		if p.End0 != path[i][1] || p.End1 != path[i][0] {
			t.Error("path error")
		}
	}
}

func Test_DTWPathWindow2(t *testing.T) {
	var dist = dtw.NewDTWWindow(s2, s1, space, 1)
	for i, p := range dist.Path() {
		if p.End0 != path1[i][1] || p.End1 != path1[i][0] {
			t.Error("path error")
		}
	}
}

func Test_DTWDBA1(t *testing.T) {
	var dist = dtw.NewDTW(s1, s2, space)
	AssertSeriesAlmostEqual(t, dba, dist.DBA(1, 1))
}

func Test_DTWDBAWindow1(t *testing.T) {
	var dist = dtw.NewDTWWindow(s1, s2, space, 1)
	AssertSeriesAlmostEqual(t, dba1, dist.DBA(1, 1))
}

func Test_DTWDBA2(t *testing.T) {
	var dist = dtw.NewDTW(s2, s1, space)
	AssertSeriesAlmostEqual(t, dba, dist.DBA(1, 1))
}

func Test_DTWDBAWindow2(t *testing.T) {
	var dist = dtw.NewDTWWindow(s2, s1, space, 1)
	AssertSeriesAlmostEqual(t, dba1, dist.DBA(1, 1))
}

func Test_DTWDBA3(t *testing.T) {
	var dist = dtw.NewDTW(s1, s2, space)
	AssertSeriesAlmostEqual(t, dbaw, dist.DBA(1, 2))
}

func Test_DTWDBAWindow3(t *testing.T) {
	var dist = dtw.NewDTWWindow(s1, s2, space, 1)
	AssertSeriesAlmostEqual(t, dbaw1, dist.DBA(2, 1))
}
