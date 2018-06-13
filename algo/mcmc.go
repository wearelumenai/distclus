package algo

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"distclus/core"
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
	// Centers Initializer
	Initializer Initializer
	// Random source seed
	Seed uint64
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
	Data    []core.Elemt
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
	return proposal.Loss(m.Data, m.config.Space, m.config.Norm)
}

func (m *MCMC) Push(elemt core.Elemt) {
	m.Data = append(m.Data, elemt)
}

func (m *MCMC) Centroids() (Clust, error) {
	var clust Clust
	var err error

	switch m.status {
	case Created:
		err = fmt.Errorf("no Clust available")
	default:
		clust = m.cur
	}

	return clust, err
}

func (m *MCMC) Predict(elemt core.Elemt, push bool) (core.Elemt, int, error) {
	var pred core.Elemt
	var idx int
	var err error

	switch m.status {
	case Created:
		err = fmt.Errorf("no Clust available")
	default:
		var clust, _ = m.Centroids()
		pred, idx = clust.UAssign(elemt, m.config.Space)
	}

	if push {
		m.Push(elemt)
	}

	return pred, idx, err
}

// Make an iterate for a proposal running with kmeans
func (m *MCMC) iterate(k int, clust *Clust) {
	var km = NewKMeans(k, 1, m.config.Space, clust.Initializer)

	km.Data = m.Data

	km.Run()
	km.Close()

	clust = &km.clust
}

// Alter a proposal using MCMC distribution
func (m *MCMC) alter(clust *Clust) {
	for i, p := range *clust {
		(*clust)[i] = m.distrib.Sample(p)
	}
}

// Compute probability between two proposals using MCMC distribution
func (m *MCMC) proba(x, mu Clust) (p float64) {
	p = 1.0
	for i := range x {
		p *= m.distrib.Pdf(x[i], mu[i])
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
	MCMCLoop(m, m.iterate, m.loss)
}

func MCMCLoop(m *MCMC, iterate func(k int, clust *Clust), loss func(proposal Clust) float64) {
	var curK = m.config.InitK
	m.cur = m.initialize(curK)

	var curLoss = m.loss(m.cur)
	var curPdf = m.proba(m.cur, m.cur)

	for i := 0; i < m.config.McmcIter; i++ {
		var propK = m.nextK(curK)
		var propCenters = m.GetCenters(propK, m.cur)

		iterate(propK, &propCenters)
		var memo = make(Clust, len(propCenters))
		copy(memo, propCenters)
		m.alter(&propCenters)

		var propLoss = loss(propCenters)
		var propPdf = m.proba(memo, propCenters)

		if m.accept(propLoss, curLoss, propPdf, curPdf, propK, curK) {
			curK = propK
			m.cur = propCenters
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
func (m *MCMC) GetCenters(k int, prev Clust) Clust {
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

// Generate a configuration of k centers based on previous configuration
func (m *MCMC) genCenters(k int, prev Clust) (clust Clust) {
	var err error
	var prevK = len(prev)

	if prevK < k {
		clust = KmeansPPIter(prev, m.Data, m.config.Space, m.src)
	}

	if prevK > k {
		var del = m.src.Intn(prevK)
		var centers = prev
		clust = append(centers[:del], centers[del+1:]...)
	}

	if prevK == k {
		clust = prev
	}

	if err != nil {
		panic(err)
	}

	return clust
}

// initialise a configuration of k centers
func (m *MCMC) initialize(k int) Clust {
	var km = NewKMeans(k, m.config.InitIter, m.config.Space, m.config.Initializer)

	for _, elemt := range m.Data {
		km.Push(elemt)
	}

	km.Run()
	km.Close()

	var clust, _ = km.Centroids()
	return clust
}

// Compute next centers number based on ProbaK
func (m *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[WeightedChoice(m.config.ProbaK(), m.src)]

	if newK < 1 {
		return 1
	}

	return newK
}
