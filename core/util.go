package core

import "sync"

func Par(process func([]Elemt, int, int, int), data []Elemt, degree int) {
	var wg = &sync.WaitGroup{}
	var offset = len(data) / degree
	var remainder = len(data) % degree
	wg.Add(degree)
	var launch = func(part []Elemt, start int, end int, rank int) {
		defer wg.Done()
		process(part, start, end, rank)
	}
	var start = 0
	for i := 0; i < degree; i++ {
		if i == degree-remainder {
			offset += 1
		}
		var end = start + offset
		if end > len(data) {
			end = len(data)
		}
		var part = data[start:end]
		go launch(part, start, end, i)
		start = end
	}
	wg.Wait()
}
