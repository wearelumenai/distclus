package series_test

import (
	"distclus/series"
	"testing"
)

func TestSeriesDist(t *testing.T) {
	t.Error("Test doest not exist")
}

func TestSeriesCombine(t *testing.T) {
	t.Error("Test doest not exist")
}

func TestSpace_Copy(t *testing.T) {
	var e1 = series.Series{{2}, {1}}
	sp := series.Space{}
	var e2 = sp.Copy(e1).(series.Series)

	if e1[0][0] != e2[0][0] || e1[1][0] != e2[1][0] {
		t.Error("Expected same elements")
	}

	e2[0][0] = 3.
	e2[1][0] = 6.

	if e1[0][0] == e2[0][0] || e1[1][0] == e2[1][0] {
		t.Error("Expected different elements")
	}
}
