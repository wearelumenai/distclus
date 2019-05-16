package cosinus

import (
	"distclus/core"
	"distclus/vectors"
	"math"
)

type Space struct {
	vspace vectors.Space
}

func NewSpace(conf Conf) Space {
	return Space{
		vspace: vectors.NewSpace(conf.Conf),
	}
}

func (Space) Dist(elemt1, elemt2 core.Elemt) float64 {
	var v1 = elemt1.([]float64)
	var v2 = elemt2.([]float64)
	return 1 - Cosinus(v1, v2)
}

func (space Space) Combine(elemt1 core.Elemt, weight1 int, elemt2 core.Elemt, weight2 int) core.Elemt {
	return space.vspace.Combine(elemt1, weight1, elemt2, weight2)
}

func (space Space) Copy(elemt core.Elemt) core.Elemt {
	return space.vspace.Copy(elemt)
}

func (space Space) Dim(data []core.Elemt) int {
	return space.vspace.Dim(data)
}

func Cosinus(v1, v2 []float64) (cos float64) {
	cos = ScalarProduct(v1, v2) / Norm(v1) / Norm(v2)
	return
}

func Norm(v []float64) float64 {
	return math.Sqrt(ScalarProduct(v, v))
}

func ScalarProduct(v1, v2 []float64) (product float64) {
	for i := range v1 {
		product += v1[i] * v2[i]
	}
	return
}
