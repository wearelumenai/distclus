package streaming

import (
	"distclus/core"
	"distclus/kmeans"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Streaming algorithm implementation
type Streaming struct {
	template    *core.AlgorithmTemplate
	data        *core.DataBuffer
	config      StreamingConf
	strategy    StreamingStrategy
	initializer core.Initializer
	uniform     distuv.Uniform
	distrib     StreamingDistrib
	store       CenterStore
	iter, acc   int
}

// StreamingStrategy interface
type StreamingStrategy interface {
	Iterate(core.Clust, int) core.Clust
	Loss(core.Clust) float64
}

func (streaming *Streaming) Centroids() (c core.Clust, err error) {
	return streaming.template.Centroids()
}

func (streaming *Streaming) Push(elemt core.Elemt) (err error) {
	return streaming.template.Push(elemt)
}

func (streaming *Streaming) Predict(elemt core.Elemt, push bool) (pred core.Elemt, label int, err error) {
	return streaming.template.Predict(elemt, push)
}

func (streaming *Streaming) Run(async bool) {
	streaming.template.Run(async)
}

func (streaming *Streaming) Close() {
	streaming.template.Close()
}

func (streaming *Streaming) initializeAlgorithm() (centroids core.Clust, ready bool) {
	streaming.data.Apply()
	return streaming.initializer(streaming.config.InitK, streaming.data.Data, streaming.config.Space, streaming.config.RGen)
}

func (streaming *Streaming) runAlgorithm(closing <-chan bool) {
	var current = proposal{
		k:       streaming.config.InitK,
		centers: streaming.template.Clust,
		loss:    streaming.strategy.Loss(streaming.template.Clust),
		pdf:     streaming.proba(streaming.template.Clust, streaming.template.Clust),
	}

	for i, loop := 0, true; i < streaming.config.StreamingIter && loop; i++ {
		select {
		case <-closing:
			loop = false

		default:
			current = streaming.doIter(current)
			streaming.data.Apply()
		}
	}
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (streaming *Streaming) doIter(current proposal) proposal {

	var prop = streaming.propose(current)

	if streaming.accept(current, prop) {
		current = prop
		streaming.template.Clust = prop.centers
		streaming.store.SetCenters(streaming.template.Clust)
		streaming.acc += 1
	}

	streaming.iter += 1
	return current
}

func (streaming *Streaming) propose(current proposal) (prop proposal) {
	prop = proposal{}
	prop.k = streaming.nextK(current.k)
	var centers = streaming.store.GetCenters(prop.k, streaming.template.Clust)
	centers = streaming.strategy.Iterate(centers, 1)
	prop.centers = streaming.alter(centers)
	prop.loss = streaming.strategy.Loss(prop.centers)
	prop.pdf = streaming.proba(prop.centers, centers)
	return
}

func (streaming *Streaming) accept(current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = streaming.config.L2B() * float64(streaming.config.Dim*(current.k-prop.k))
	var rGibbs = streaming.config.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return streaming.uniform.Rand() < rho
}

func (streaming *Streaming) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(streaming.config.ProbaK, streaming.config.RGen)]

	switch {
	case newK < 1:
		return 1
	case newK > streaming.config.MaxK:
		return streaming.config.MaxK
	case newK > len(streaming.data.Data):
		return len(streaming.data.Data)
	default:
		return newK
	}
}

func (streaming *Streaming) alter(clust core.Clust) (result core.Clust) {
	result = make(core.Clust, len(clust))

	for i := range clust {
		result[i] = streaming.distrib.Sample(clust[i])
	}

	return
}

func (streaming *Streaming) proba(x, mu core.Clust) (p float64) {
	p = 0.
	for i := range x {
		p += streaming.distrib.Pdf(mu[i], x[i])
	}
	return p
}

func (streaming *Streaming) AcceptRatio() float64 {
	return float64(streaming.acc) / float64(streaming.iter)
}
