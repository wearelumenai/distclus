package clustering_go

import (
	"fmt"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

type MCMCDistrib interface {
	Sample(mu Elemt) Elemt
	Pdf(x, mu Elemt) float64
	Init(dim int, tau, nu float64, src rand.Source) bool
}

type MCMCConf struct {
	dim                int
	frameSize          int
	b, amp, lamb, tau  float64
	norm, nu           float64
	initK              int
	mcmcIter, initIter int
	probaK             []float64
	space              space
	initializer        Initializer
	seed               uint64
}

func NewMCMCConf(dim, frameSize int, b, amp, norm, nu float64, initK, mcmcIter, initIter int,
	space space, initializer Initializer, seed uint64) MCMCConf {
	var c MCMCConf
	c.dim = dim
	c.frameSize = frameSize
	c.b = b
	c.amp = amp
	c.lamb = c.amp * math.Sqrt(float64(dim+3)/float64(frameSize))
	c.tau = 1 / math.Sqrt(float64(frameSize*20))
	c.norm = norm
	c.nu = nu
	c.initIter = initIter
	c.initK = initK
	c.mcmcIter = mcmcIter
	c.probaK = []float64{1, 8, 1}
	c.space = space
	c.initializer = initializer
	c.seed = seed
	return c
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

func NewMCMC(conf MCMCConf, distrib MCMCDistrib) MCMC {
	var m MCMC
	m.config = conf
	m.store = make(map[int]Clust)
	m.status = Created
	m.distrib = distrib
	m.src = rand.New(rand.NewSource(conf.seed))
	ok := distrib.Init(conf.dim, conf.tau, conf.nu, m.src)
	if !ok {
		panic("can't initialize MCMCDistrib")
	}
	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.src}
	return m
}

func (m *MCMC) loss(proposal Clust) float64 {
	return proposal.Loss(&m.data, m.config.space, m.config.norm)
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
		var idx = assign(elemt, *m.cur.Centers(), m.config.space)
		return m.cur.Center(idx), idx, nil
	}
}

func (m *MCMC) iterate(k int, proposal Clust) Clust {
	var initializer = func(k2 int, elemts []Elemt, space space) (Clust, error) {
		return proposal, nil
	}
	var km = NewKMeans(k, 1, m.config.space, initializer)
	for i := range m.data {
		km.Push(m.data[i])
	}
	km.Run()
	km.Close()
	var clust, _ = km.Centroids()
	return clust
}

func (m *MCMC) alter(proposal Clust) Clust {
	var res = make([]Elemt, len(proposal.centers))
	for i, p := range proposal.centers {
		res[i] = m.distrib.Sample(p)
	}
	var c, _ = NewClustering(res)
	return c
}

func (m *MCMC) proba(proposal1, proposal2 Clust) (p float64) {
	for i, c1 := range proposal1.centers {
		var c2 = proposal2.centers[i]
		p *= m.distrib.Pdf(c1, c2)
	}
	return p
}

func (m *MCMC) accept(pLoss, cLoss float64, pPdf, cPdf float64, pK, cK int) bool {
	// adjust lambda to avoid very large gibbs measure
	if m.config.lamb*pLoss > 50 {
		m.config.lamb = 50 / pLoss
	}

	var rProp = cPdf / pPdf
	var rInit = math.Pow(2*m.config.b, float64(m.config.dim*(cK-pK)))
	var rGibbs = math.Exp(-m.config.lamb * (pLoss - cLoss))

	var rho = rGibbs * rInit * rProp
	return m.uniform.Rand() < rho
}

func (m *MCMC) Run() {
	var curK = m.config.initK
	m.cur = m.initialize(curK)
	var curLoss = m.loss(m.cur)
	var curPdf = m.proba(m.cur, m.cur)
	for i := 0; i < m.config.mcmcIter; i++ {
		var propK = m.nextK(curK)
		var propCenters = m.getCenters(propK, curK)
		propCenters = m.iterate(propK, propCenters)
		var prop = m.alter(propCenters)
		var propLoss = m.loss(prop)
		var propPdf = m.proba(prop, propCenters)
		if m.accept(propLoss, curLoss, propPdf, curPdf, propK, curK){
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

func (m *MCMC) getCenters(k, prevK int) Clust {
	var centers, ok = m.store[k]
	if !ok {
		m.store[k] = m.genCenters(k, prevK)
		centers, _ = m.store[k]
	}
	return centers
}

func (m *MCMC) setCenters(clust Clust) {
	m.store[len(clust.centers)] = clust
}

func (m *MCMC) genCenters(k, prevK int) Clust {
	return m.initialize(k)
}

func (m *MCMC) initialize(k int) Clust {
	var km = NewKMeans(k, m.config.initIter, m.config.space, m.config.initializer)
	for _, elemt := range m.data {
		km.Push(elemt)
	}
	km.Run()
	km.Close()
	var clust, _ = km.Centroids()
	return clust
}

func (m *MCMC) nextK(k int) int {
	var prob = m.config.probaK
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
