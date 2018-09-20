package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"runtime"
	"sync"
)

func NewParMCMC(conf MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {
	var algo = NewSeqMCMC(conf, distrib, initializer, data)
	algo.MCMCSupport = ParMCMCSupport{config: conf, buffer: &algo.Buffer, space:algo.Space, norm: algo.Norm}
	return algo
}

type ParMCMCSupport struct {
	config MCMCConf
	buffer *core.Buffer
	space  core.Space
	norm float64
}

func (support ParMCMCSupport) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{K: len(clust), Iter: iter, Space: support.config.Space}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, support.buffer.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()
	return result
}

func (support ParMCMCSupport) Loss(clust core.Clust) float64 {
	var out = support.startMCMCWorkers(clust)
	var aggr = lossAggregate(out)
	return aggr.sum
}

func (support ParMCMCSupport) startMCMCWorkers(clust core.Clust) (chan msgMCMC) {
	var degree = runtime.NumCPU()
	var offset = (len(support.buffer.Data)-1)/degree + 1
	var out = make(chan msgMCMC, degree)
	var wg = &sync.WaitGroup{}

	wg.Add(degree)
	for i := 0; i < degree; i++ {
		var part = getChunk(i, offset, support.buffer.Data)
		go lossMapReduce(clust, part, support.space, support.norm, out, wg)
	}

	wg.Wait()
	close(out)

	return out
}

func getChunk(i int, offset int, elemts []core.Elemt) []core.Elemt {
	var start = i * offset
	var end = start + offset

	if end > len(elemts) {
		end = len(elemts)
	}

	return elemts[start:end]
}

type msgMCMC struct {
	sum float64
	card int
}

func lossMapReduce(clust core.Clust, elemts []core.Elemt, space core.Space, norm float64, out chan<- msgMCMC, wg *sync.WaitGroup) {
	defer wg.Done()

	var reduced msgMCMC
	reduced.sum = clust.Loss(elemts, space, norm)
	reduced.card = len(elemts)

	out <- reduced
}

func lossAggregate(out chan msgMCMC) msgMCMC {
	var aggregate msgMCMC
	for agg := range out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
