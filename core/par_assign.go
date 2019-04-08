package core

func parMapLabel(centroids Clust, data []Elemt, space Space, degree int) []int {
	var result = make([]int, len(data))

	var process = func(start int, end int, rank int) {
		copy(result[start:end], centroids.MapLabel(data[start:end], space))
	}

	Par(process, len(data), degree)
	return result
}
