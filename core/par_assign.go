package core

func parMapLabel(centroids Clust, data []Elemt, space Space, degree int) []int {
	var result = make([]int, len(data))

	var process = func(part []Elemt, start int, end int, rank int) {
		copy(result[start:end], centroids.MapLabel(part, space))
	}

	Par(process, data, degree)
	return result
}
