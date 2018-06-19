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
	B, Amp, lamb, tau float64
	// Loss normalisation coefficient
	Norm float64
	// Degrees of freedom
	Nu float64
	// Initial number of centers
	InitK int
	// Iteration numbers for mcmc and centers initialisation
	McmcIter, InitIter int
	probaK             []float64
	// Space where Data are include
	Space core.Space
	// Random source seed
	RGen *rand.Rand
}

// Probability for next K (-1, 0, +1)
func (c *MCMCConf) ProbaK() []float64 {
	if len(c.probaK) == 0 {
		c.probaK = []float64{1, 8, 1}
	}
	return c.probaK
}

func (c *MCMCConf) Tau() float64 {
	if c.tau == 0 {
		c.tau = 1 / math.Sqrt(float64(c.FrameSize*20))
	}
	return c.tau
}

func (c *MCMCConf) Lamb() float64 {
	if c.lamb == 0 {
		c.lamb = c.Amp * math.Sqrt(float64(c.Dim+3)/float64(c.FrameSize))
	}
	return c.lamb
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
	lamb        float64
	lb          float64
	closing     chan bool
	closed      chan bool
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
	m.lamb = conf.Lamb()
	m.lb = math.Log(2*m.B)

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		m.rgen = rand.New(rand.NewSource(seed))
	} else {
		m.rgen = conf.RGen
	}

	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.rgen}
	m.closing = make(chan bool, 1)
	m.closed = make(chan bool, 1)

	return m
}

func (m *MCMC) Centroids() (c Clust, err error) {
	switch m.status {
	case Created:
		err = fmt.Errorf("clustering not started")
	default:
		c = m.clust
	}

	return
}

func (m *MCMC) Push(elemt core.Elemt) (err error) {
	switch m.status {
	case Closed:
		err = errors.New("clustering ended")
	default:
		m.Data = append(m.Data, elemt)
	}

	return err
}

func (m *MCMC) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int

	var clust, err = m.Centroids()

	if err == nil {
		pred, idx, _ = clust.Assign(elemt, m.Space)
		if push {
			err = m.Push(elemt)
		}
	}

	return pred, idx, err
}

func (m *MCMC) Run(async bool) {
	m.status = Running

	var init = m.initializer(m.InitK, m.Data, m.MCMCConf.Space, m.rgen)
	m.clust = m.Iterate(*m, init, m.InitIter)

	if async {
		go m.process()
	} else {
		m.process()
	}
}

func (m *MCMC) Close() {
	m.closing <- true
	<-m.closed
}

func (m *MCMC) process() {
	var curK = m.InitK
	var curLoss = m.Loss(*m, m.clust)
	var curPdf = m.proba(m.clust, m.clust)

	for i, loop := 0, true; i < m.McmcIter && loop; i++ {
		select {
		case <-m.closing:
			loop = false

		default:
			var propK = m.nextK(curK)
			var propCenters = m.getCenters(propK, m.clust)

			propCenters = m.Iterate(*m, propCenters, 1)

			var prop = m.alter(propCenters)

			var propLoss = m.Loss(*m, prop)
			var propPdf = m.proba(prop, propCenters)

			if m.accept(propLoss, curLoss, propPdf, curPdf, propK, curK) {
				curK = propK
				m.clust = prop
				curLoss = propLoss
				curPdf = propPdf
				m.setCenters(m.clust)
			}
		}
	}

	m.status = Closed
	m.closed <- true
}

// Alter a proposal using MCMC distribution
func (m *MCMC) alter(clust Clust) Clust {
	var result = make(Clust, len(clust))

	for i := range clust {
		result[i] = m.distrib.Sample(clust[i])
	}

	return result
}

// Compute probability between two proposals using MCMC distribution
func (m *MCMC) proba(x, mu Clust) (p float64) {
	p = 0.
	for i := range x {
		p += m.distrib.Pdf(mu[i], x[i])
	}
	return p
}

// Compute acceptance of a proposal(p* parameters) against a current proposal(c* parameters) using loss, pdf and K
func (m *MCMC) accept(pLoss, cLoss float64, pPdf, cPdf float64, pK, cK int) bool {
	// adjust lambda to avoid very large gibbs measure

	var rProp = cPdf - pPdf
	var rInit = m.lb * float64(m.Dim*(cK-pK))
	var rGibbs = -m.lamb * (pLoss - cLoss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return m.uniform.Rand() < rho
}

// Compute next centers number based on ProbaK
func (m *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[WeightedChoice(m.ProbaK(), m.rgen)]

	if newK < 1 {
		return 1
	}

	return newK
}

// Get a configuration center(retrieve from store if K is exist else create with genCenters
func (m *MCMC) getCenters(k int, prev Clust) Clust {
	var centers, ok = m.store[k]

	if !ok {
		centers = m.genCenters(k, prev)
		m.store[k] = centers
	}

	return centers
}

// Set a configuration in store
func (m *MCMC) setCenters(clust Clust) {
	m.store[len(clust)] = clust
}

// Generate a configuration of K centers based on previous configuration
func (m *MCMC) genCenters(k int, prev Clust) (clust Clust) {
	var prevK = len(prev)

	switch {
	case prevK < k:
		clust = make(Clust, k)
		for i := 0; i < prevK; i++ {
			clust[i] = m.Space.Copy(prev[i])
		}
		clust[k-1] = KmeansPPIter(prev, m.Data, m.Space, m.rgen)

	case prevK > k:
		var del = m.rgen.Intn(prevK)
		clust = make(Clust, k)
		for i := 0; i < k; i++ {
			if i < del {
				clust[i] = m.Space.Copy(prev[i])
			} else {
				clust[i] = m.Space.Copy(prev[i+1])
			}
		}

	case prevK == k:
		clust = prev
	}

	return
}
