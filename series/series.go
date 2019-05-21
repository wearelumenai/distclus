package series

import (
	"distclus/core"
)

// Space for processing vectors of vectors ([][]float64)
type Space struct {
	window     int
	innerSpace core.Space
}

// NewSpace create a new series space
func NewSpace(conf core.SpaceConf) Space {
	var sconf = conf.(Conf)
	return Space{
		window:     sconf.Window,
		innerSpace: sconf.InnerSpace,
	}
}

func (space Space) Dist(elemt1, elemt2 core.Elemt) (sum float64) {
	var s1, s2 = space.getSeries(elemt1, elemt2)
	var dtw = NewDTWWindow(s1, s2, space.innerSpace, space.window)
	return dtw.Dist()
}

func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) core.Elemt {
	var s1, s2 = space.getSeries(elemt1, elemt2)
	var dtw = NewDTWWindow(s1, s2, space.innerSpace, space.window)
	return dtw.DBA(weight1, weight2)
}

func (space Space) getSeries(elemt1 core.Elemt, elemt2 core.Elemt) ([][]float64, [][]float64) {
	var e1 = elemt1.([][]float64)
	var e2 = elemt2.([][]float64)
	return ShrinkLongest(e1, e2, space.innerSpace, space.window)
}

// Copy creates a copy of a vector
func (space Space) Copy(elemt core.Elemt) core.Elemt {
	var rv = elemt.([][]float64)
	var copied = make([][]float64, len(rv))
	for i := range copied {
		copied[i] = make([]float64, len(rv[i]))
		copy(copied[i], rv[i])
	}
	return copied
}

// Dim returns input data dimension
func (space Space) Dim(data []core.Elemt) (dim int) {
	if len(data) > 0 {
		series := data[0].([][]float64)
		if len(series) > 0 {
			dim = len(series[0])
		}
	}
	return
}
