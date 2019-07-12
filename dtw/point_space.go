package dtw

import "distclus/core"

// PointSpace represents a space for points in R
type PointSpace interface {
	core.Space
	PointDist(point1 []float64, point2 []float64) float64
	PointCombine(point1 []float64, weight1 int, point2 []float64, weight2 int) []float64
	PointCopy(point []float64) []float64
}
