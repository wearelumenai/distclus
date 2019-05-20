package series

import (
	"distclus/core"
)

func Interpolate(ts [][]float64, idx []int, shrinkFactor int, space core.Space) [][]float64 {
	var last = idx[len(ts)-1] / shrinkFactor
	if idx[len(ts)-1]%shrinkFactor > 0 {
		last++
	}
	var result = make([][]float64, last)
	result[0] = ts[0]
	for i, j := 1, 1; i < last; i++ {
		var x = i * shrinkFactor
		for ; x > idx[j]; j++ {
		}
		if idx[j] == x {
			result[i] = ts[j]
		} else {
			result[i] = space.Combine(ts[j-1], idx[j]-x, ts[j], x-idx[j-1]).([]float64)
		}
	}
	return result
}

func Resize(ts [][]float64, size int, space core.Space) [][]float64 {
	var idx = make([]int, len(ts))
	for i := range idx {
		idx[i] = i * (size - 1)
	}
	var resized = Interpolate(ts, idx, len(ts)-1, space)
	return append(resized, ts[len(ts)-1])
}
