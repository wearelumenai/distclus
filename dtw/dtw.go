package dtw

// DTW represents the Dynamic Time Warping distance between 2 series
type DTW struct {
	s1, s2 [][]float64
	space  PointSpace
	window int
	path   [][]int
	dist   float64
}

// NewDTW creates a new DTW instance
func NewDTW(s1, s2 [][]float64, space PointSpace) DTW {
	return NewDTWWindow(s1, s2, space, 0)
}

// NewDTWWindow creates a new DTW instance constrained to the given window
func NewDTWWindow(s1, s2 [][]float64, space PointSpace, window int) DTW {
	var dtw = DTW{
		s1:     s1,
		s2:     s2,
		space:  space,
		window: window,
	}
	var cost = NewCumCostMatrix(s1, s2, space, window)
	dtw.path = cost.computePath()
	dtw.dist = cost.Get(len(cost.s1)-1, len(cost.s2)-1)
	return dtw
}

// Path returns the minimal cost path computed by Dynamic Time Warping
func (dtw DTW) Path() [][]int {
	return dtw.path
}

// Dist returns the Dynamic Time Warping distance value
func (dtw DTW) Dist() float64 {
	return dtw.dist
}

// DBA computes the average between series with the given weights
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
	return Interpolate(ts, idx, shrinkFactor, dtw.space)
}
