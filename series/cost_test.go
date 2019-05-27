package series_test

import (
	"distclus/series"
	"distclus/vectors"
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
var cumCost1 = [][]float64{
	{1, 1, inf, inf, inf, inf, inf},
	{2, 1, 1, inf, inf, inf, inf},
	{inf, 2, 2, 1, inf, inf, inf},
	{inf, inf, 4, 2, 1, inf, inf},
	{inf, inf, inf, 2, 2, 3, inf},
	{inf, inf, inf, inf, 5, 6, 5},
}

func Test_CumCost1(t *testing.T) {
	var dtw = series.NewCumCostMatrix(s1, s2, space, 0)
	for i1 := range s1 {
		for i2 := range s2 {
			if dtw.Get(i1, i2) != cumCost[i1][i2] {
				t.Error("distance error for", i1, i2)
			}
		}
	}
}

func Test_CumCost2(t *testing.T) {
	var dtw = series.NewCumCostMatrix(s2, s1, space, 0)
	for i2 := range s2 {
		for i1 := range s1 {
			if dtw.Get(i2, i1) != cumCost[i1][i2] {
				t.Error("distance error for", i2, i1)
			}
		}
	}
}

func Test_CumCostWindow1(t *testing.T) {
	var dtw = series.NewCumCostMatrix(s1, s2, space, 1)
	for i1 := range s1 {
		for i2 := range s2 {
			if dtw.Get(i1, i2) != cumCost1[i1][i2] {
				t.Error("distance error for", i1, i2)
				return
			}
		}
	}
}

func Test_CumCostWindow2(t *testing.T) {
	var dtw = series.NewCumCostMatrix(s2, s1, space, 1)
	for i2 := range s2 {
		for i1 := range s1 {
			if dtw.Get(i2, i1) != cumCost1[i1][i2] {
				t.Error("distance error for", i2, i1)
				return
			}
		}
	}
}
