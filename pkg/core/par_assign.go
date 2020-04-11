package core

func parMapLabel(centroids Clust, data []Elemt, space Space, degree int) (labels []int, dists []float64) {
	labels = make([]int, len(data))
	dists = make([]float64, len(data))

	var process = func(start int, end int, rank int) {
		var partLabels, partDists = centroids.MapLabel(data[start:end], space)
		copy(labels[start:end], partLabels)
		copy(dists[start:end], partDists)
	}

	Par(process, len(data), degree)
	return
}
