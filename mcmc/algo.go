package mcmc

import (
	"distclus/core"
	"distclus/kmeans"
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"time"
)


type MCMC struct {
	MCMCSupport
	core.Buffer
	config MCMCConf
	store       CenterStore
	distrib     MCMCDistrib
	initializer core.Initializer
	uniform     distuv.Uniform
	clust       core.Clust
	status      core.ClustStatus
	closing     chan bool
	closed      chan bool
	iter, acc   int
}

type MCMCSupport interface {
	Iterate(core.Clust, int) core.Clust
	Loss(core.Clust) float64
}

type MCMCDistrib interface {
	Sample(mu core.Elemt) core.Elemt
	Pdf(x, mu core.Elemt) float64
}

func NewSeqMCMC(conf MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {

	conf.Verify()
	setConfigDefaults(&conf)

	var m MCMC
	m.config = conf
	m.status = core.Created
	m.initializer = initializer
	m.distrib = distrib
	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.config.RGen}
	m.closing = make(chan bool, 1)
	m.closed = make(chan bool, 1)
	m.Buffer = core.NewBuffer(data, m.config.FrameSize)
	m.MCMCSupport = SeqMCMCSupport{buffer:&m.Buffer, config:m.config}
	m.store = NewCenterStore(&m.Buffer, conf.Space, m.config.RGen)
	return &m
}

func setConfigDefaults(conf *MCMCConf) {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		conf.RGen = rand.New(rand.NewSource(seed))
	}
	if len(conf.ProbaK) == 0 {
		conf.ProbaK = []float64{1, 0, 9}
	}
	if conf.MaxK == 0 {
		conf.MaxK = 16
	}
}

func (mcmc *MCMC) Centroids() (c core.Clust, err error) {
	switch mcmc.status {
	case core.Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = mcmc.clust
	}

	return
}

func (mcmc *MCMC) Push(elemt core.Elemt) (err error) {
	switch mcmc.status {
	case core.Closed:
		err = errors.New("clustering ended")
	default:
		mcmc.Buffer.Push(elemt)
	}

	return err
}

func (mcmc *MCMC) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int

	var clust, err = mcmc.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, mcmc.config.Space)
		if push {
			err = mcmc.Push(elemt)
		}
	}

	return pred, idx, err
}

func (mcmc *MCMC) Run(async bool) {
	if async {
		mcmc.Buffer.SetAsync()
		go mcmc.initAndRun(async)
	} else {
		mcmc.initAndRun(async)
	}
}

func (mcmc *MCMC) initAndRun(async bool) {
	for ok := false; !ok; {
		mcmc.clust, ok = mcmc.initializer(mcmc.config.InitK,
			mcmc.Data, mcmc.config.Space, mcmc.config.RGen)
		if !ok {
			mcmc.handleFailedInitialisation(async)
		}
	}
	mcmc.status = core.Running
	mcmc.runAlgorithm()
}

func (mcmc *MCMC) handleFailedInitialisation(async bool) {
	if !async {
		panic("failed to initialize")
	}
	time.Sleep(300 * time.Millisecond)
	mcmc.Buffer.Apply()
}

func (mcmc *MCMC) runAlgorithm() {
	var current = proposal{
		k:    mcmc.config.InitK,
		loss: mcmc.Loss(mcmc.clust),
		pdf:  mcmc.proba(mcmc.clust, mcmc.clust),
	}

	for i, loop := 0, true; i < mcmc.config.McmcIter && loop; i++ {
		select {
		case <-mcmc.closing:
			loop = false

		default:
			current = mcmc.doIter(current)
			mcmc.Apply()
		}
	}

	mcmc.status = core.Closed
	mcmc.closed <- true
}

func (mcmc *MCMC) doIter(current proposal) proposal {

	var prop = mcmc.propose(current)

	if mcmc.accept(current, prop) {
		current = prop
		mcmc.clust = prop.centers
		mcmc.store.SetCenters(mcmc.clust)
		mcmc.acc += 1
	}

	mcmc.iter += 1
	return current
}

func (mcmc *MCMC) propose(current proposal) proposal {
	var prop proposal
	prop.k = mcmc.nextK(current.k)
	var centers = mcmc.store.GetCenters(prop.k, mcmc.clust)
	centers = mcmc.Iterate(centers, 1)
	prop.centers = mcmc.alter(centers)
	prop.loss = mcmc.Loss(prop.centers)
	prop.pdf = mcmc.proba(prop.centers, centers)
	return prop
}

func (mcmc *MCMC) accept(current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = mcmc.config.L2B() * float64(mcmc.config.Dim*(current.k-prop.k))
	var rGibbs = mcmc.config.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return mcmc.uniform.Rand() < rho
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (mcmc *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(mcmc.config.ProbaK, mcmc.config.RGen)]

	switch {
	case newK < 1:
		return 1
	case newK > mcmc.config.MaxK:
		return mcmc.config.MaxK
	case newK > len(mcmc.Data):
		return len(mcmc.Data)
	default:
		return newK
	}
}

func (mcmc *MCMC) alter(clust core.Clust) core.Clust {
	var result = make(core.Clust, len(clust))

	for i := range clust {
		result[i] = mcmc.distrib.Sample(clust[i])
	}

	return result
}

func (mcmc *MCMC) proba(x, mu core.Clust) (p float64) {
	p = 0.
	for i := range x {
		p += mcmc.distrib.Pdf(mu[i], x[i])
	}
	return p
}

func (mcmc *MCMC) Close() {
	mcmc.closing <- true
	<-mcmc.closed
}

func (mcmc *MCMC) AcceptRatio() float64 {
	return float64(mcmc.acc) / float64(mcmc.iter)
}

type SeqMCMCSupport struct {
	config MCMCConf
	buffer *core.Buffer
}

func (support SeqMCMCSupport) Iterate(clust core.Clust, iter int) core.Clust {
	var conf = kmeans.KMeansConf{
		K: len(clust),
		Iter: iter,
		Space: support.config.Space,
		RGen: support.config.RGen,
	}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, support.buffer.Data)

	km.Run(false)
	km.Close()
	var result, _ = km.Centroids()

	return result
}

func (support SeqMCMCSupport) Loss(proposal core.Clust) float64 {
	return proposal.Loss(support.buffer.Data, support.config.Space, support.config.Norm)
}
