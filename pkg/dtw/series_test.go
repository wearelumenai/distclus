package dtw_test

import (
	"lumenai.fr/v0/distclus/pkg/dtw"
	"testing"
)

var conf = dtw.Conf{
	Window:     1,
	InnerSpace: space,
}

func TestSpace_Copy(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var check = func(vectors [][]float64) {
		var copied = space.Copy(vectors).([][]float64)
		if len(vectors) != len(copied) {
			t.Error("array length is different")
		}
		for i, vector := range vectors {
			if len(vector) != len(copied[i]) {
				t.Error("array sub length is different")
			}
			for j, data := range vector {
				if data != copied[i][j] {
					t.Error("value is different")
				}
			}
		}
	}
	check(cumCost)
	check(cumCost1)
}

func TestSpace_Dist(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var dist = space.Dist(s1, s2)
	if dist != 5 {
		t.Error("dist error")
	}
}

func TestSpace_DistShrink(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var s21 = dtw.Resize(s2, 13, conf.InnerSpace)
	var dist = space.Dist(s1, s21)
	if dist != 5 {
		t.Error("dist error")
	}
}

func TestSpace_Combine(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var s = space.Combine(s1, 1, s2, 1)
	AssertSeriesAlmostEqual(t, dba1, s.([][]float64))
}

func TestSpace_CombineShrink(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var s21 = dtw.Resize(s2, 13, conf.InnerSpace)
	var s = space.Combine(s1, 1, s21, 1)
	AssertSeriesAlmostEqual(t, dba1, s.([][]float64))
}

func TestSpace_CombineWeight(t *testing.T) {
	var space = dtw.NewSpace(conf)
	var s = space.Combine(s1, 2, s2, 1)
	AssertSeriesAlmostEqual(t, dbaw1, s.([][]float64))
}
