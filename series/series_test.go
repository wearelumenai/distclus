package series_test

import (
	"distclus/series"
	"testing"
)

var conf = series.Conf{
	Window:     1,
	InnerSpace: space,
}

func TestSpace_Dist(t *testing.T) {
	var space = series.NewSpace(conf)
	var dist = space.Dist(s1, s2)
	if dist != 5 {
		t.Error("dist error")
	}
}

func TestSpace_DistShrink(t *testing.T) {
	var space = series.NewSpace(conf)
	var s21 = series.Resize(s2, 13, conf.InnerSpace)
	var dist = space.Dist(s1, s21)
	if dist != 5 {
		t.Error("dist error")
	}
}

func TestSpace_Combine(t *testing.T) {
	var space = series.NewSpace(conf)
	var s = space.Combine(s1, 1, s2, 1)
	AssertSeriesAlmostEqual(t, dba1, s.([][]float64))
}

func TestSpace_CombineShrink(t *testing.T) {
	var space = series.NewSpace(conf)
	var s21 = series.Resize(s2, 13, conf.InnerSpace)
	var s = space.Combine(s1, 1, s21, 1)
	AssertSeriesAlmostEqual(t, dba1, s.([][]float64))
}

func TestSpace_CombineWeight(t *testing.T) {
	var space = series.NewSpace(conf)
	var s = space.Combine(s1, 2, s2, 1)
	AssertSeriesAlmostEqual(t, dbaw1, s.([][]float64))
}
