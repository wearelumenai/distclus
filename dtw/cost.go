package dtw

import (
	"distclus/core"
	"math"
)

// CumCostMatrix represents the accumulated cost matrix needed to compute DTW distance.
type CumCostMatrix struct {
	s1, s2    [][]float64
	values    []float64
	stride    []int
	space     core.Space
	window    int
	transpose bool
}

// NewCumCostMatrix creates at new CumCostMatrix instance.
func NewCumCostMatrix(s1, s2 [][]float64, space core.Space, window int) CumCostMatrix {
	var cost = CumCostMatrix{
		s1:     s1,
		s2:     s2,
		space:  space,
		window: window,
	}
	var l1, l2 = len(s1), len(s2)
	if l1 < l2 {
		cost.transpose = false
		cost.setStride(l1, l2)
	} else {
		cost.transpose = true
		cost.setStride(l2, l1)
	}
	cost.computeCumCost()
	return cost
}

// Get returns the accumulated cost at position (i1,i2)
func (cumCost *CumCostMatrix) Get(i1, i2 int) float64 {
	var i = cumCost.ravel(i1, i2)
	if i == -1 {
		return math.Inf(1)
	}
	return cumCost.values[i]
}

func (cumCost *CumCostMatrix) setStride(shortest int, longest int) {
	if cumCost.window == 0 {
		cumCost.window = longest
	}
	var maxWindow = 2*cumCost.window + 1
	if maxWindow < longest {
		cumCost.stride = []int{shortest, maxWindow}
	} else {
		cumCost.stride = []int{shortest, longest}
	}
}

func (cumCost *CumCostMatrix) ravel(i1, i2 int) int {
	if !cumCost.inWindow(i1, i2) {
		return -1
	}
	if cumCost.transpose {
		return cumCost.shift(i2, i1)
	}
	return cumCost.shift(i1, i2)
}

func (cumCost *CumCostMatrix) inWindow(i int, j int) bool {
	return i-j <= cumCost.window && j-i <= cumCost.window
}

func (cumCost *CumCostMatrix) shift(i, j int) int {
	if i > cumCost.window {
		return cumCost.index(i, j-i+cumCost.window)
	}
	return cumCost.index(i, j)
}

func (cumCost *CumCostMatrix) index(i, j int) int {
	return i*cumCost.stride[1] + j
}

func (cumCost *CumCostMatrix) computeCumCost() {
	cumCost.values = make([]float64, cumCost.stride[0]*cumCost.stride[1])
	for i1 := range cumCost.s1 {
		for i2 := range cumCost.s2 {
			var i = cumCost.ravel(i1, i2)
			if i >= 0 {
				var cost = 0.
				var dist = cumCost.space.Dist(cumCost.s1[i1], cumCost.s2[i2])
				switch {
				case i1 == 0 && i2 == 0:
					cost = dist
				case i1 == 0:
					cost = cumCost.Get(i1, i2-1) + dist
				case i2 == 0:
					cost = cumCost.Get(i1-1, i2) + dist
				default:
					var l, u, ul = cumCost.neighborCosts(i1, i2)
					cost = min(ul, u, l) + dist
				}
				cumCost.values[i] = cost
			}
		}
	}
}

func (cumCost *CumCostMatrix) computePath() [][]int {
	var path = make([][]int, 0)
	var i1, i2 = len(cumCost.s1) - 1, len(cumCost.s2) - 1
	for i1 > 0 || i2 > 0 {
		path = append(path, []int{i1, i2})
		switch {
		case i1 == 0:
			i2--
		case i2 == 0:
			i1--
		default:
			var l, u, ul = cumCost.neighborCosts(i1, i2)
			i1, i2 = decrPath(ul, u, l, i1, i2)
		}
	}
	path = append(path, []int{0, 0})
	return path
}

func (cumCost *CumCostMatrix) neighborCosts(i1 int, i2 int) (float64, float64, float64) {
	var l = cumCost.Get(i1-1, i2)
	var u = cumCost.Get(i1, i2-1)
	var ul = cumCost.Get(i1-1, i2-1)
	return l, u, ul
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
