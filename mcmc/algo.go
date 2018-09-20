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

type MCMCConf struct {
	Dim                int
	FrameSize          int
	B, Amp, R          float64
	Norm               float64
	Nu                 float64
	InitK              int
	MaxK               int
	McmcIter, InitIter int
	ProbaK             []float64
	Space              core.Space
	RGen               *rand.Rand
	lamb, l2b, tau     float64
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

func (conf *MCMCConf) verify() {
	if conf.InitK < 1 {
		panic(fmt.Sprintf("Illegal value for K: %v", conf.InitK))
	}

	if conf.InitK > conf.MaxK && conf.MaxK != 0 {
		panic(fmt.Sprintf("Illegal value for Max K / Init K: %v / %v", conf.MaxK, conf.InitK))
	}

	if conf.McmcIter < 0 {
		panic(fmt.Sprintf("Illegal value for Iter: %v", conf.McmcIter))
	}
}

func (conf *MCMCConf) getRGen() *rand.Rand {
	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		return rand.New(rand.NewSource(seed))
	} else {
		return conf.RGen
	}
}

type MCMCDistrib interface {
	Sample(mu core.Elemt) core.Elemt
	Pdf(x, mu core.Elemt) float64
}

type MCMCSupport interface {
	Iterate(core.Clust, int) core.Clust
	Loss(core.Clust) float64
}

type MCMC struct {
	MCMCConf
	MCMCSupport
	core.Buffer
	distrib     MCMCDistrib
	initializer core.Initializer
	uniform     distuv.Uniform
	store       map[int]core.Clust
	clust       core.Clust
	status      core.ClustStatus
	rgen        *rand.Rand
	closing     chan bool
	closed      chan bool
	iter, acc   int
}

func NewSeqMCMC(conf MCMCConf, distrib MCMCDistrib, initializer core.Initializer, data []core.Elemt) *MCMC {

	conf.verify()

	var m MCMC
	m.MCMCConf = conf
	m.store = make(map[int]core.Clust)
	m.status = core.Created
	m.initializer = initializer
	m.distrib = distrib
	m.rgen = conf.getRGen()
	m.uniform = distuv.Uniform{Max: 1, Min: 0, Src: m.rgen}
	m.closing = make(chan bool, 1)
	m.closed = make(chan bool, 1)
	m.Buffer = core.NewBuffer(data, m.FrameSize)
	m.MCMCSupport = SeqMCMCSupport{buffer:&m.Buffer, norm:m.Norm, space:m.Space, rgen:m.RGen}

	if len(m.ProbaK) == 0 {
		m.ProbaK = []float64{1, 0, 9}
	}

	if m.MaxK == 0 {
		m.MaxK = 16
	}

	return &m
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
		pred, idx, _ = clust.Assign(elemt, mcmc.Space)
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
		mcmc.clust, ok = mcmc.initializer(mcmc.InitK,
			mcmc.Data, mcmc.MCMCConf.Space, mcmc.rgen)
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
		k:    mcmc.InitK,
		loss: mcmc.Loss(mcmc.clust),
		pdf:  mcmc.proba(mcmc.clust, mcmc.clust),
	}

	for i, loop := 0, true; i < mcmc.McmcIter && loop; i++ {
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
		mcmc.setCenters(mcmc.clust)
		mcmc.acc += 1
	}

	mcmc.iter += 1
	return current
}

type proposal struct {
	k       int
	centers core.Clust
	loss    float64
	pdf     float64
}

func (mcmc *MCMC) propose(current proposal) proposal {
	var prop proposal
	prop.k = mcmc.nextK(current.k)
	var centers = mcmc.getCenters(prop.k, mcmc.clust)
	centers = mcmc.Iterate(centers, 1)
	prop.centers = mcmc.alter(centers)
	prop.loss = mcmc.Loss(prop.centers)
	prop.pdf = mcmc.proba(prop.centers, centers)
	return prop
}

func (mcmc *MCMC) accept(current proposal, prop proposal) bool {
	var rProp = current.pdf - prop.pdf
	var rInit = mcmc.L2B() * float64(mcmc.Dim*(current.k-prop.k))
	var rGibbs = mcmc.Lambda() * (current.loss - prop.loss)

	var rho = math.Exp(rGibbs + rInit + rProp)
	return mcmc.uniform.Rand() < rho
}

func (mcmc *MCMC) nextK(k int) int {
	var newK = k + []int{-1, 0, 1}[kmeans.WeightedChoice(mcmc.ProbaK, mcmc.rgen)]

	switch {
	case newK < 1:
		return 1
	case newK > mcmc.MaxK:
		return mcmc.MaxK
	case newK > len(mcmc.Data):
		return len(mcmc.Data)
	default:
		return newK
	}
}

func (mcmc *MCMC) getCenters(k int, clust core.Clust) core.Clust {
	var centers, ok = mcmc.store[k]

	if !ok {
		centers = mcmc.genCenters(k, clust)
		mcmc.store[k] = centers
	}

	return centers
}

func (mcmc *MCMC) setCenters(clust core.Clust) {
	mcmc.store[len(clust)] = clust
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

func (mcmc *MCMC) genCenters(k int, prev core.Clust) core.Clust {
	var prevK = len(prev)
	var clust core.Clust

	switch {
	case prevK < k:
		clust = mcmc.addCenter(prevK, prev)

	case prevK > k:
		clust = mcmc.delCenter(prevK, prev)

	case prevK == k:
		clust = prev
	}

	return clust
}

func (mcmc *MCMC) addCenter(prevK int, prev core.Clust) core.Clust {
	var clust = make(core.Clust, prevK+1)
	for i := 0; i < prevK; i++ {
		clust[i] = mcmc.Space.Copy(prev[i])
	}
	clust[prevK] = kmeans.KMeansPPIter(prev, mcmc.Data, mcmc.Space, mcmc.rgen)
	return clust
}

func (mcmc *MCMC) delCenter(prevK int, prev core.Clust) core.Clust {
	var del = mcmc.rgen.Intn(prevK)
	var clust = make(core.Clust, prevK-1)
	for i := 0; i < prevK-1; i++ {
		if i < del {
			clust[i] = mcmc.Space.Copy(prev[i])
		} else {
			clust[i] = mcmc.Space.Copy(prev[i+1])
		}
	}
	return clust
}

func (mcmc *MCMC) Close() {
	mcmc.closing <- true
	<-mcmc.closed
}

func (mcmc *MCMC) AcceptRatio() float64 {
	return float64(mcmc.acc) / float64(mcmc.iter)
}

type SeqMCMCSupport struct {
	buffer *core.Buffer
	space  core.Space
	norm float64
	rgen   *rand.Rand
}

func (support SeqMCMCSupport) Iterate(clust core.Clust, iter int) core.Clust {
	conf := kmeans.KMeansConf{len(clust), iter, support.space, support.rgen}
	var km = kmeans.NewSeqKMeans(conf, clust.Initializer, support.buffer.Data)

	km.Run(false)
	km.Close()

	var result, _ = km.Centroids()

	return result
}

func (support SeqMCMCSupport) Loss(proposal core.Clust) float64 {
	return proposal.Loss(support.buffer.Data, support.space, support.norm)
}
