package series

import (
	"distclus/core"
)

func Interpolate(s [][]float64, idx []int, shrinkFactor int, space core.Space) [][]float64 {
	var last = idx[len(s)-1] / shrinkFactor
	if idx[len(s)-1]%shrinkFactor > 0 {
		last++
	}
	var result = make([][]float64, last)
	result[0] = s[0]
	for i, j := 1, 1; i < last; i++ {
		var x = i * shrinkFactor
		for ; x > idx[j]; j++ {
		}
		if idx[j] == x {
			result[i] = s[j]
		} else {
			result[i] = space.Combine(s[j-1], idx[j]-x, s[j], x-idx[j-1]).([]float64)
		}
	}
	return result
}

func Resize(s [][]float64, size int, space core.Space) [][]float64 {
	var idx = make([]int, len(s))
	for i := range idx {
		idx[i] = i * (size - 1)
	}
	var resized = Interpolate(s, idx, len(s)-1, space)
	return append(resized, s[len(s)-1])
}

func ShrinkLongest(s1, s2 [][]float64, space core.Space, window int) (sl1, sl2 [][]float64) {
	var l1, l2 = len(s1), len(s2)
	switch {
	case l1 > l2+window:
		sl1 = Resize(s1, l2+window, space)
		sl2 = s2
	case l2 > l1+window:
		sl1 = s1
		sl2 = Resize(s2, l1+window, space)
	default:
		sl1 = s1
		sl2 = s2
	}
	return
}
