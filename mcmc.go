package clustering_go

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

// MCMC distribution interface
type MCMCDistrib interface {
	// Sample from a distribution with mu expectancy
	Sample(mu Elemt) Elemt
	// Density from a distribution with mu expectancy and x point
	Pdf(x, mu Elemt) float64
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
	Nu                 float64
	// Initial number of centers
	InitK              int
	// Iteration numbers for mcmc and centers initialisation
	McmcIter, InitIter int
	probaK             []float64
	// Space where data are include
	Space              space
	// Centers initializer
	Initializer        Initializer
	// Random source seed
	Seed               uint64
}

// Probability for next k (-1, 0, +1)
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
	config  MCMCConf
	distrib MCMCDistrib
	uniform distuv.Uniform
	store   map[int]Clust
	data    []Elemt
	cur     Clust
	status  ClustStatus
	src     *rand.Rand
}

// Constructor for MCMC
func NewMCMC(conf MCMCConf, distrib MCMCDistrib) MCMC {
	var m MCMC
	m.config = conf
	m.store = make(map[int]Clust)
	m.status = Created
	m.distrib = distrib
	m.src = rand.New(rand.NewSource(conf.Seed))
	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.src}
	return m
}

// Compute loss proposal based on Clust.Loss
func (m *MCMC) loss(proposal Clust) float64 {
	return proposal.Loss(&m.data, m.config.Space, m.config.Norm)
}

func (m *MCMC) Push(elemt Elemt) {
	m.data = append(m.data, elemt)
}

func (m *MCMC) Centroids() (c Clust, err error) {
	switch m.status {
	case Created:
		err = fmt.Errorf("no Clust available")
	default:
		c = m.cur
	}
	return c, err
}

func (m *MCMC) Predict(elemt Elemt) (c Elemt, idx int, err error) {
	switch m.status {
	case Created:
		return c, idx, fmt.Errorf("no Clust available")
	default:
		c, idx = m.cur.UAssign(elemt, m.config.Space)
		return c, idx, nil
	}
}

// Make an iteration for a proposal running with kmeans
func (m *MCMC) iterate(k int, proposal Clust) Clust {
	var initializer = func(k2 int, elemts []Elemt, space space) (Clust, error) {
		return proposal, nil
	}
	var km = NewKMeans(k, 1, m.config.Space, initializer)
	for i := range m.data {
		km.Push(m.data[i])
	}
	km.Run()
	km.Close()
	var clust, _ = km.Centroids()
	return clust
}

// Alter a proposal using MCMC distribution
func (m *MCMC) alter(proposal Clust) Clust {
	var res = make([]Elemt, len(proposal.centers))
	for i, p := range proposal.centers {
		res[i] = m.distrib.Sample(p)
	}
	var c, _ = NewClustering(res)
	return c
}

// Compute probability between two proposals using MCMC distribution
func (m *MCMC) proba(proposal1, proposal2 Clust) (p float64) {
	for i, c1 := range proposal1.centers {
		var c2 = proposal2.centers[i]
		p *= m.distrib.Pdf(c1, c2)
	}
	return p
}

// Compute acceptance of a proposal(p* parameters) against a current proposal(c* parameters) using loss, pdf and k
func (m *MCMC) accept(pLoss, cLoss float64, pPdf, cPdf float64, pK, cK int) bool {
	// adjust lambda to avoid very large gibbs measure
	var lamb = m.config.Lamb()
	if lamb*pLoss > 50 {
		lamb = 50 / pLoss
	}

	var rProp = cPdf / pPdf
	var rInit = math.Pow(2*m.config.B, float64(m.config.Dim*(cK-pK)))
	var rGibbs = math.Exp(-lamb * (pLoss - cLoss))

	var rho = rGibbs * rInit * rProp
	return m.uniform.Rand() < rho
}

func (m *MCMC) Run() {
	var curK = m.config.InitK
	m.cur = m.initialize(curK)
	var curLoss = m.loss(m.cur)
	var curPdf = m.proba(m.cur, m.cur)
	for i := 0; i < m.config.McmcIter; i++ {
		var propK = m.nextK(curK)
		var propCenters = m.getCenters(propK, curK)
		propCenters = m.iterate(propK, propCenters)
		var prop = m.alter(propCenters)
		var propLoss = m.loss(prop)
		var propPdf = m.proba(prop, propCenters)
		if m.accept(propLoss, curLoss, propPdf, curPdf, propK, curK) {
			curK = propK
			m.cur = prop
			curLoss = propLoss
			curPdf = propPdf
			m.setCenters(m.cur)
		}
	}
}

func (m *MCMC) Close() {
	m.status = Closed
}

// Get a configuration center(retrieve from store if k is exist else create with genCenters
func (m *MCMC) getCenters(k, prevK int) Clust {
	var centers, ok = m.store[k]
	if !ok {
		m.store[k] = m.genCenters(k, prevK)
		centers, _ = m.store[k]
	}
	return centers
}

// Set a configuration in store
func (m *MCMC) setCenters(clust Clust) {
	m.store[len(clust.centers)] = clust
}

// Generate a configuration of k centers based on previous configuration
func (m *MCMC) genCenters(k, prevK int) Clust {
	return m.initialize(k)
}

// initialise a configuration of k centers
func (m *MCMC) initialize(k int) Clust {
	var km = NewKMeans(k, m.config.InitIter, m.config.Space, m.config.Initializer)
	for _, elemt := range m.data {
		km.Push(elemt)
	}
	km.Run()
	km.Close()
	var clust, _ = km.Centroids()
	return clust
}

// Compute next centers number based on ProbaK
func (m *MCMC) nextK(k int) int {
	var prob = m.config.ProbaK()
	var less, same, more = prob[0], prob[1], prob[2]
	var sum = less + same + more
	var proba = m.src.Float64() * sum
	var res = 0
	if proba > less+same {
		res = 1
	}
	if proba < less {
		res = -1
	}
	return k + res
}
