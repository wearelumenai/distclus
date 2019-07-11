package dtw

import (
	"math"
)

type Ends struct {
	End0, End1 int
}

// CumCostMatrix represents the accumulated cost matrix needed to compute DTW distance.
type CumCostMatrix struct {
	s1, s2           [][]float64
	values           []float64
	space            PointSpace
	window           int
	stride0, stride1 int
	path             []Ends
}

// NewCumCostMatrix creates at new CumCostMatrix instance.
func NewCumCostMatrix(s1, s2 [][]float64, space PointSpace, window int) CumCostMatrix {
	var cost = CumCostMatrix{
		s1:     s1,
		s2:     s2,
		space:  space,
		window: window,
	}
	var l1, l2 = len(s1), len(s2)
	cost.setStride(l1, l2)
	cost.computeCumCost()
	cost.computePath()
	return cost
}

// Get returns the accumulated cost at position (i1,i2)
func (cumCost *CumCostMatrix) Get(i1, i2 int) float64 {
	if !cumCost.inWindow(i1, i2) {
		return math.Inf(1)
	}
	var i = cumCost.ravel(i1, i2)
	return cumCost.values[i]
}

func (cumCost *CumCostMatrix) Path() []Ends {
	return cumCost.path
}

func (cumCost *CumCostMatrix) setStride(shortest int, longest int) {
	if cumCost.window == 0 {
		cumCost.window = longest
	}
	var maxWindow = 2*cumCost.window + 1
	cumCost.stride0 = shortest
	if maxWindow < longest {
		cumCost.stride1 = maxWindow
	} else {
		cumCost.stride1 = longest
	}
}

func (cumCost *CumCostMatrix) computeCumCost() {
	var space = cumCost.space
	cumCost.values = make([]float64, cumCost.stride0*cumCost.stride1)
	for i1 := range cumCost.s1 {
		var i2l = 0
		if !cumCost.inWindow(i1, i2l) {
			i2l = i1 - cumCost.window
		}
		var i2r = len(cumCost.s2) - 1
		if !cumCost.inWindow(i1, i2r) {
			i2r = i1 + cumCost.window
		}
		for i2 := i2l; i2 <= i2r; i2++ {
			var i = cumCost.ravel(i1, i2)
			var cost = 0.
			var dist = space.PointDist(cumCost.s1[i1], cumCost.s2[i2])
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

func (cumCost *CumCostMatrix) computePath() {
	var i1, i2 = len(cumCost.s1) - 1, len(cumCost.s2) - 1
	cumCost.path = make([]Ends, 0, i1+i2)
	for i1 > 0 || i2 > 0 {
		cumCost.path = append(cumCost.path, Ends{i1, i2})
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
	cumCost.path = append(cumCost.path, Ends{0, 0})
}

func (cumCost *CumCostMatrix) neighborCosts(i1 int, i2 int) (float64, float64, float64) {
	var l = cumCost.Get(i1-1, i2)
	var u = cumCost.Get(i1, i2-1)
	var ul = cumCost.Get(i1-1, i2-1)
	return l, u, ul
}

func (cumCost *CumCostMatrix) ravel(i, j int) int {
	if i > cumCost.window {
		return cumCost.index(i, j-i+cumCost.window)
	}
	return cumCost.index(i, j)
}

func (cumCost *CumCostMatrix) index(i, j int) int {
	return i*cumCost.stride1 + j
}

func (cumCost *CumCostMatrix) inWindow(i int, j int) bool {
	return i-j <= cumCost.window && j-i <= cumCost.window
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
