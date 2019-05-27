package core

import "sync"

// PartitionProcess represents a function that runs in parallel over a data partitions
type PartitionProcess = func(start, end, rank int)

// Par runs a function in parallel over data partitions given data size and degree of parallelism
func Par(process func(int, int, int), size int, degree int) {
	var wg = &sync.WaitGroup{}
	var offset = size / degree
	var remainder = size % degree
	wg.Add(degree)
	var launch = func(start int, end int, rank int) {
		defer wg.Done()
		process(start, end, rank)
	}
	var start = 0
	for i := 0; i < degree; i++ {
		if i == degree-remainder {
			offset++
		}
		var end = start + offset
		if end > size {
			end = size
		}
		go launch(start, end, i)
		start = end
	}
	wg.Wait()
}
