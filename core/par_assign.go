package core

import "sync"

type assignWorker struct {
	result []int
	wg     *sync.WaitGroup
}

func parMapLabel(centroids Clust, data []Elemt, space Space, degree int) []int {
	var offset = (len(data)-1)/degree + 1
	var workers = assignWorker{
		result: make([]int, len(data)),
		wg:     &sync.WaitGroup{},
	}
	workers.wg.Add(degree)

	for i := 0; i < degree; i++ {
		var part = GetChunk(i, offset, data)
		go workers.assignMap(space, centroids, part, i*offset)
	}

	workers.wg.Wait()
	return workers.result
}

func (strategy *assignWorker) assignMap(space Space, centroids Clust, elemts []Elemt, offset int) {
	defer strategy.wg.Done()
	copy(strategy.result[offset:], centroids.MapLabel(elemts, space))
}
