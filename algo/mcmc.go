package algo

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"distclus/core"
	"time"
	"errors"
)

// MCMC distribution interface
type MCMCDistrib interface {
	// Sample from a distribution with mu expectancy
	Sample(mu core.Elemt) core.Elemt
	// Density from a distribution with mu expectancy and x point
	Pdf(x, mu core.Elemt) float64
}

type MCMCConf struct {
	// Data dimension
	Dim int
	// Number of element to retain in the partition
	FrameSize int
	// Mcmc parameters
	B, Amp, R float64
	// Loss normalisation coefficient
	Norm float64
	// Degrees of freedom
	Nu float64
	// Initial number of centers
	InitK int
	// Iteration numbers for mcmc and centers initialisation
	McmcIter, InitIter int
	ProbaK             []float64
	// Space where Data are include
	Space core.Space
	// Random source seed
	RGen *rand.Rand
	// memoize
	lamb, l2b, tau float64
}

func (conf *MCMCConf) Tau() float64 {
	if conf.tau == 0 {
		conf.tau = 1 / math.Sqrt(float64(conf.FrameSize*20))
	}
	return conf.tau
}

func (c *MCMCConf) L2B() float64 {
	if c.l2b == 0 {
		c.l2b = math.Log(2 * c.B)
	}
	return c.l2b
}

func (conf *MCMCConf) Lambda() float64 {
	if conf.lamb == 0 {
		var r = conf.R

		if r == 0 {
			r = 1
		}

		conf.lamb = conf.Amp * math.Sqrt(float64(conf.Dim+3)/float64(conf.FrameSize))
	}

	return conf.lamb
}

type MCMC struct {
	MCMCConf
	MCMCSupport
	Data    []core.Elemt
	distrib MCMCDistrib
	// Centers Initializer
	initializer Initializer
	uniform     distuv.Uniform
	store       map[int]Clust
	clust       Clust
	status      ClustStatus
	rgen        *rand.Rand
	closing     chan bool
	closed      chan bool
	iter, acc   int
}

type MCMCSupport interface {
	Iterate(MCMC, Clust, int) Clust
	Loss(MCMC, Clust) float64
}

type SeqMCMCSupport struct {
}

func (SeqMCMCSupport) Iterate(m MCMC, clust Clust, iter int) Clust {
	conf := KMeansConf{len(clust), iter, m.Space, m.rgen}
	var km = NewKMeans(conf, clust.Initializer)

	km.Data = m.Data

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()

	return result
}

func (SeqMCMCSupport) Loss(m MCMC, proposal Clust) float64 {
	return proposal.Loss(m.Data, m.Space, m.Norm)
}

// Constructor for MCMC
func NewMCMC(conf MCMCConf, distrib MCMCDistrib, initializer Initializer) MCMC {

	if conf.InitK < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.InitK))
	}

	if conf.McmcIter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.McmcIter))
	}

	var m MCMC
	m.MCMCConf = conf
	m.MCMCSupport = SeqMCMCSupport{}
	m.store = make(map[int]Clust)
	m.status = Created
	m.initializer = initializer
	m.distrib = distrib

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		m.rgen = rand.New(rand.NewSource(seed))
	} else {
		m.rgen = conf.RGen
	}

	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.rgen}
	m.closing = make(chan bool, 1)
	m.closed = make(chan bool, 1)

	if len(m.ProbaK) == 0 {
		m.ProbaK = []float64{1, 0, 9}
	}

	return m
}

func (mcmc *MCMC) AcceptRatio() float64 {
	return float64(mcmc.acc) / float64(mcmc.iter)
}

func (mcmc *MCMC) Centroids() (c Clust, err error) {
	switch mcmc.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = mcmc.clust
	}

	return
}

func (mcmc *MCMC) Push(elemt core.Elemt) (err error) {
	switch mcmc.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		mcmc.Data = append(mcmc.Data, elemt)
	}

	return err
}

func (mcmc *MCMC) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int

	var clust, err = mcmc.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, mcmc.Space)
		if push {
			err = mcmc.Push(elemt)
		}
	}

	return pred, idx, err
}

func (mcmc *MCMC) Run(async bool) {
	mcmc.status = Running

	var init = mcmc.initializer(mcmc.InitK, mcmc.Data, mcmc.MCMCConf.Space, mcmc.rgen)
	mcmc.clust = mcmc.Iterate(*mcmc, init, mcmc.InitIter)

	if async {
		go mcmc.process()
	} else {
		mcmc.process()
	}
}

func (mcmc *MCMC) Close() {
	mcmc.closing <- true
	<-mcmc.closed
}

func (mcmc *MCMC) process() {
	var curK = mcmc.InitK
	var curLoss = mcmc.Loss(*mcmc, mcmc.clust)
	var curPdf = mcmc.proba(mcmc.clust, mcmc.clust)

	for i, loop := 0, true; i < mcmc.McmcIter && loop; i++ {
		select {
		case <-mcmc.closing:
			loop = false

		default:
			var propK = mcmc.nextK(curK)
			var propCenters = mcmc.getCenters(propK, mcmc.clust)

			propCenters = mcmc.Iterate(*mcmc, propCenters, 1)

			var prop = mcmc.alter(propCenters)

			var propLoss = mcmc.Loss(*mcmc, prop)
			var propPdf = mcmc.proba(prop, propCenters)

			if mcmc.accept(propLoss, curLoss, propPdf, curPdf, propK, curK) {
				curK = propK
				mcmc.clust = prop
				curLoss = propLoss
				curPdf = propPdf
				mcmc.setCenters(mcmc.clust)
				mcmc.acc += 1
			}
			mcmc.iter += 1
		}
	}

	mcmc.status = Closed
	mcmc.closed <- true
}

// Alter a proposal using MCMC distribution
func (mcmc *MCMC) alter(clust Clust) Clust {
	var result = make(Clust, len(clust))

	for i := range clust {
		result[i] = mcmc.distrib.Sample(clust[i])
	}

	return result
}

// Compute probability between two proposals using MCMC distribution
func (mcmc *MCMC) proba(x, mu Clust) (p float64) {
	p = 0.
	for i := range x {
		p += mcmc.distrib.Pdf(mu[i], x[i])
	}
	return p
}

// Compute acceptance of a proposal(p* parameters) against a current proposal(c* parameters) using loss, pdf and K
func (mcmc *MCMC) accept(pLoss, cLoss float64, pPdf, cPdf float64, pK, cK int) bool {
	// adjust lambda to avoid very large gibbs measure

	var rProp = cPdf - pPdf
	var rInit = mcmc.L2B() * float64(mcmc.Dim*(cK-pK))
	var rGibbs = -mcmc.Lambda() * (pLoss - cLoss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return mcmc.uniform.Rand() < rho
}

// Compute next centers number based on ProbaK
func (mcmc *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[WeightedChoice(mcmc.ProbaK, mcmc.rgen)]

	if newK < 1 {
		return 1
	}

	return newK
}

// Get a configuration center(retrieve from store if K is exist else create with genCenters
func (mcmc *MCMC) getCenters(k int, prev Clust) Clust {
	var centers, ok = mcmc.store[k]

	if !ok {
		centers = mcmc.genCenters(k, prev)
		mcmc.store[k] = centers
	}

	return centers
}

// Set a configuration in store
func (mcmc *MCMC) setCenters(clust Clust) {
	mcmc.store[len(clust)] = clust
}

// Generate a configuration of K centers based on previous configuration
func (mcmc *MCMC) genCenters(k int, prev Clust) (clust Clust) {
	var prevK = len(prev)

	switch {
	case prevK < k:
		clust = make(Clust, k)
		for i := 0; i < prevK; i++ {
			clust[i] = mcmc.Space.Copy(prev[i])
		}
		clust[k-1] = KmeansPPIter(prev, mcmc.Data, mcmc.Space, mcmc.rgen)

	case prevK > k:
		var del = mcmc.rgen.Intn(prevK)
		clust = make(Clust, k)
		for i := 0; i < k; i++ {
			if i < del {
				clust[i] = mcmc.Space.Copy(prev[i])
			} else {
				clust[i] = mcmc.Space.Copy(prev[i+1])
			}
		}

	case prevK == k:
		clust = prev
	}

	return
}
