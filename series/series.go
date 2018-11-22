package series

import (
	"distclus/complex"
	"distclus/core"
	"distclus/real"
	"strings"
)

// Space for processing reals of reals ([][]float64)
type Space struct {
	window     int
	innerSpace core.Space
}

// Series of multi-float dimensions
type Series [][]float64

// NewSpace create a new series space
func NewSpace(conf core.Conf) Space {
	var innerSpace core.Space
	var sconf = conf.(Conf)
	switch strings.ToLower(sconf.InnerSpace) {
	case "complex":
		innerSpace = complex.NewSpace(conf)
	default:
		innerSpace = real.NewSpace(conf)
	}
	return Space{
		window:     sconf.Window,
		innerSpace: innerSpace,
	}
}

func getMatrix(elemt1, elemt2 core.Elemt) (matrix Series) {
	var realSpace = real.Space{}

	var e1 = elemt1.(Series)
	var e2 = elemt2.(Series)

	matrix = make(Series, len(e1))

	for i1, el1 := range e1 {
		matrix[i1] = make([]float64, len(e2))
		for i2, el2 := range e2 {
			matrix[i1][i2] = realSpace.Dist(el1, el2)
		}
	}

	return matrix
}

func fillMatrix(e1, e2 Series, matrix Series) (sum float64) {
	var i2 = 0

	for i1 := range e1 {
		sum += matrix[i1][i2]
		if i2 == len(e2) {
			for i1++; i1 < len(e1); i1++ {
				sum += matrix[i1][i2]
			}
			break
		} else if i1 == len(e1)-1 {
			for i2++; i2 < len(e2); i2++ {
				sum += matrix[i1][i2]
			}
			break
		} else if matrix[i1+1][i2] > matrix[i1][i2+1] {
			i2++
		}
	}
	return
}

// Dist computes euclidean distance between two nodes
func (space Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var e1 = elemt1.(Series)
	var e2 = elemt2.(Series)

	var matrix = getMatrix(elemt1, elemt2)

	return fillMatrix(e1, e2, matrix)
}

// Combine computes combination between two nodes
func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) {

}

// Copy creates a copy of a vector
func (space Space) Copy(elemt core.Elemt) core.Elemt {
	var rv = elemt.(Series)
	var copied = make(Series, len(rv))
	for i := range copied {
		copied[i] = make([]float64, len(rv[i]))
		copy(copied[i], rv[i])
	}
	return copied
}
