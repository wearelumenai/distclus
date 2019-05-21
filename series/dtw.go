package series

import (
	"distclus/core"
)

type DTW struct {
	s1, s2 [][]float64
	space  core.Space
	window int
	path   [][]int
	dist   float64
}

func NewDTW(s1, s2 [][]float64, space core.Space) DTW {
	return NewDTWWindow(s1, s2, space, 0)
}

func NewDTWWindow(s1, s2 [][]float64, space core.Space, window int) DTW {
	var dtw = DTW{
		s1:     s1,
		s2:     s2,
		space:  space,
		window: window,
	}
	var cost = NewCumCost(s1, s2, space, window)
	dtw.path = cost.computePath()
	dtw.dist = cost.Get(len(cost.s1)-1, len(cost.s2)-1)
	return dtw
}

func (dtw DTW) Path() [][]int {
	return dtw.path
}

func (dtw DTW) Dist() float64 {
	return dtw.dist
}

func (dtw DTW) DBA(w1, w2 int) [][]float64 {
	var dba = make([][]float64, len(dtw.path))
	var idx = make([]int, len(dtw.path))
	for i := range dtw.path {
		var ends = dtw.path[len(dtw.path)-1-i]
		dba[i] = dtw.space.Combine(dtw.s1[ends[0]], w1, dtw.s2[ends[1]], w2).([]float64)
		idx[i] = ends[0]*w1 + ends[1]*w2
	}
	return dtw.interpolate(dba, idx, w1+w2)
}

func (dtw *DTW) interpolate(ts [][]float64, idx []int, shrinkFactor int) [][]float64 {
	result := Interpolate(ts, idx, shrinkFactor, dtw.space)
	return result
}
