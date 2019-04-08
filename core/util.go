package core

import "sync"

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
			offset += 1
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
