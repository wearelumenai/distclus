package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"sync"
)

// NewParImpl returns a new parallelized algorithm implementation
func NewParImpl(conf *Conf, initializer core.Initializer, data []core.Elemt, distrib Distrib) (impl Impl) {
	impl = NewSeqImpl(conf, initializer, data, distrib)
	impl.strategy = &ParStrategy{
		Degree: conf.NumCPU,
	}
	return
}

// ParStrategy defines a parallelized strategy
type ParStrategy struct {
	Degree int
}

type workerSupport struct {
	ParStrategy
	out chan msg
	wg  *sync.WaitGroup
}

type msg struct {
	sum  float64
	card int
}

// Iterate is the iterative execution
func (strategy *ParStrategy) Iterate(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt, iter int) core.Clust {
	var kmeansConf = core.Conf{
		ImplConf: kmeans.Conf{
			K:    len(centroids),
			Iter: iter,
		},
		SpaceConf: nil,
	}
	var algo = kmeans.NewAlgo(kmeansConf, space, data, centroids.Initializer)

	algo.Run(false)
	algo.Close()

	var result, _ = algo.Centroids()

	return result
}

// Loss aclculates the loss distance of input centroids
func (strategy *ParStrategy) Loss(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt) float64 {
	var workers = strategy.startWorkers(conf, space, centroids, data)
	var aggr = workers.lossAggregate()
	return aggr.sum
}

func (strategy *ParStrategy) startWorkers(conf Conf, space core.Space, centroids core.Clust, data []core.Elemt) workerSupport {
	var offset = (len(data)-1)/strategy.Degree + 1
	var workers = workerSupport{}
	workers.ParStrategy = *strategy
	workers.out = make(chan msg, strategy.Degree)
	workers.wg = &sync.WaitGroup{}
	workers.wg.Add(strategy.Degree)

	for i := 0; i < strategy.Degree; i++ {
		var part = core.GetChunk(i, offset, data)
		go workers.lossMapReduce(conf, space, centroids, part)
	}

	workers.wg.Wait()
	close(workers.out)

	return workers
}

func (strategy *workerSupport) lossMapReduce(conf Conf, space core.Space, centroids core.Clust, elemts []core.Elemt) {
	defer strategy.wg.Done()

	var reduced msg
	reduced.sum = centroids.Loss(elemts, space, conf.Norm)
	reduced.card = len(elemts)

	strategy.out <- reduced
}

func (strategy *workerSupport) lossAggregate() msg {
	var aggregate msg
	for agg := range strategy.out {
		aggregate.sum += agg.sum
		aggregate.card += agg.card
	}
	return aggregate
}
