package series

import (
	"distclus/core"
	"distclus/real"
	"math"
	"strings"
)

// Space for processing reals of reals ([][]float64)
type Space struct {
	window     int
	innerSpace core.Space
}

// NewSpace create a new series space
func NewSpace(conf core.Conf) Space {
	var innerSpace core.Space
	var sconf = conf.(Conf)
	switch strings.ToLower(sconf.InnerSpace) {
	case "real":
		innerSpace = real.NewSpace(conf)
	default:
		innerSpace = real.NewSpace(conf)
	}
	return Space{
		window:     sconf.Window,
		innerSpace: innerSpace,
	}
}

func allocate(in [][]float64, dim int) (out [][]float64) {
	out = make([][]float64, dim)
	lenin := len(in[0])
	for index := range out {
		out[index] = make([]float64, lenin)
	}
	return
}

func interpolate(in [][]float64, out [][]float64) {
	lenout := len(out)
	lenin := len(in)
	var pos float64
	for index := range out {
		if index == 0 || index == (lenout-1) {
			out[index] = in[index]
		} else {
			pos = (float64)(index * lenin / lenout)
			var ceil = math.Ceil(pos)
			var p = pos - ceil
			iindex := (int)(ceil)
			i0 := in[iindex]
			i1 := in[iindex+1]
			for ii, ii0 := range i0 {
				ii1 := i1[ii]
				out[index][ii] = (float64)((ii0 + ii1) * p)
			}
		}
	}
}

func (space Space) getMatrix(elemt1, elemt2 core.Elemt) (matrix [][]float64) {
	var innerSpace = space.innerSpace

	var e1 = elemt1.([][]float64)
	var e2 = elemt2.([][]float64)

	len1 := len(e1)
	len2 := len(e2)

	size := (int)(math.Min((float64)(len1), (float64)(len2)))

	matrix = make([][]float64, size)

	var rows [][]float64
	var cols [][]float64

	if len1 > len2 {
		rows = e2
		cols = allocate(e1, len2)
	} else if len2 > len1 {
		rows = e1
		cols = allocate(e2, len1)
	} else {
		rows = e1
		cols = e2
	}

	for rowIndex, rowVal := range rows {
		matrix[rowIndex] = make([]float64, size)
		for colIndex, colVal := range cols {
			matrix[rowIndex][colIndex] = space.innerSpace.Dist(rowVal, colVal)
		}
	}

	for i1, el1 := range e1 {
		matrix[i1] = make([]float64, len(e2))
		for i2, el2 := range e2 {
			matrix[i1][i2] = innerSpace.Dist(el1, el2)
		}
	}

	return matrix
}

func dtw(matrix [][]float64) float64 {

	for i := range matrix {
		matrix[0][i] = -1
		matrix[i][0] = -1
	}
	matrix[0][0] = 0

	for i := range matrix[1:] {
		for j := range matrix[i][1:] {
			insertion := matrix[i-1][j]
			deletion := matrix[i][j-1]
			match := matrix[i-1][j-1]
			matrix[i][j] += math.Min(math.Min(insertion, deletion), match)
		}
	}

	return matrix[len(matrix)][len(matrix)]
}

// Dist computes euclidean distance between two nodes
func (space Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var matrix = space.getMatrix(elemt1, elemt2)

	return dtw(matrix)
}

// Combine computes combination between two nodes
func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) {
	var matrix = space.getMatrix(elemt1, elemt2)
	for i := range matrix {
		for j := range matrix {
			matrix[i][j] *= (float64)(weight1 + weight2)
		}
	}
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
