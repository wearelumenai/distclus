package series

import (
	"distclus/core"
	"math"
)

type DTW struct {
	s1, s2    [][]float64
	space     core.Space
	cumCost   []float64
	path      [][]int
	window    int
	stride    []int
	transpose bool
}

func NewDTW(s1, s2 [][]float64, space core.Space) DTW {
	return NewDTWWindow(s1, s2, space, 0)
}

func NewDTWWindow(s1, s2 [][]float64, space core.Space, window int) DTW {
	var l1 = len(s1)
	var l2 = len(s2)
	var dtw = DTW{
		s1:     s1,
		s2:     s2,
		space:  space,
		window: window,
	}
	if l1 < l2 {
		dtw.transpose = false
		dtw.setStride(l1, l2)
	} else {
		dtw.transpose = true
		dtw.setStride(l2, l1)
	}
	dtw.computeCumCost()
	dtw.computePath()
	return dtw
}

func (dtw *DTW) setStride(shortest int, longest int) {
	if dtw.window == 0 {
		dtw.window = longest
	}
	var maxWindow = 2*dtw.window + 1
	if maxWindow < longest {
		dtw.stride = []int{shortest, maxWindow}
	} else {
		dtw.stride = []int{shortest, longest}
	}
}

func (dtw DTW) CumCost(i1, i2 int) float64 {
	var i = dtw.ravel(i1, i2)
	if i == -1 {
		return math.Inf(1)
	}
	return dtw.cumCost[i]
}

func (dtw DTW) Path() [][]int {
	return dtw.path
}

func (dtw DTW) Dist() float64 {
	return dtw.CumCost(len(dtw.s1)-1, len(dtw.s2)-1)
}

func (dtw DTW) ravel(i1, i2 int) int {
	if !dtw.inWindow(i1, i2) {
		return -1
	}
	if dtw.transpose {
		return dtw.shift(i2, i1)
	} else {
		return dtw.shift(i1, i2)
	}
}

func (dtw DTW) inWindow(i int, j int) bool {
	return i-j <= dtw.window && j-i <= dtw.window
}

func (dtw DTW) shift(i, j int) int {
	if i > dtw.window {
		return dtw.index(i, j-i+dtw.window)
	}
	return dtw.index(i, j)
}

func (dtw DTW) index(i, j int) int {
	return i*dtw.stride[1] + j
}

func (dtw *DTW) computeCumCost() {
	dtw.cumCost = make([]float64, dtw.stride[0]*dtw.stride[1])
	for i1 := range dtw.s1 {
		for i2 := range dtw.s2 {
			var i = dtw.ravel(i1, i2)
			if i >= 0 {
				var cost = 0.
				var dist = dtw.space.Dist(dtw.s1[i1], dtw.s2[i2])
				switch {
				case i1 == 0 && i2 == 0:
					cost = dist
				case i1 == 0:
					cost = dtw.CumCost(i1, i2-1) + dist
				case i2 == 0:
					cost = dtw.CumCost(i1-1, i2) + dist
				default:
					var l, u, ul = dtw.neighborCosts(i1, i2)
					cost = min(ul, u, l) + dist
				}
				dtw.cumCost[i] = cost
			}
		}
	}
}

func (dtw *DTW) computePath() {
	dtw.path = make([][]int, 0)
	var i1, i2 = len(dtw.s1) - 1, len(dtw.s2) - 1
	for i1 > 0 || i2 > 0 {
		dtw.path = append(dtw.path, []int{i1, i2})
		switch {
		case i1 == 0:
			i2--
		case i2 == 0:
			i1--
		default:
			var l, u, ul = dtw.neighborCosts(i1, i2)
			i1, i2 = decrPath(ul, u, l, i1, i2)
		}
	}
	dtw.path = append(dtw.path, []int{0, 0})
}

func (dtw *DTW) neighborCosts(i1 int, i2 int) (float64, float64, float64) {
	var l = dtw.CumCost(i1-1, i2)
	var u = dtw.CumCost(i1, i2-1)
	var ul = dtw.CumCost(i1-1, i2-1)
	return l, u, ul
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

func min(v1 float64, v2 float64, v3 float64) float64 {
	switch {
	case v1 <= v2 && v1 <= v3:
		return v1
	case v2 <= v3:
		return v2
	default:
		return v3
	}
}

func decrPath(ul float64, u float64, l float64, i1 int, i2 int) (int, int) {
	switch {
	case ul <= u && ul <= l:
		return i1 - 1, i2 - 1
	case l <= u:
		return i1 - 1, i2
	default:
		return i1, i2 - 1
	}
}
