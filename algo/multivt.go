package algo

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"distclus/core"
	"time"
)

// Configuration for multivariateT distribution
type MultivTConf struct {
	MCMCConf
}

// Real(float64[]) distribution wrapping StudentsT of Gonum
type MultivT struct {
	MultivTConf
	sigma mat.Symmetric
	d     *distmv.StudentsT
	rgen  *rand.Rand
}

// Constructor for multivT distribution
func NewMultivT(conf MultivTConf) MultivT {
	var sigma = make([]float64, conf.Dim*conf.Dim)
	var nextX int
	var nextY int
	var tau = conf.Tau()
	for i := range sigma {
		var x = i / conf.Dim
		var y = i % conf.Dim
		if x == nextX && y == nextY {
			sigma[i] = tau
			nextX += 1
			nextY += 1
		}
	}
	var m = MultivT{}
	m.sigma = mat.NewSymDense(conf.Dim, sigma)

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		m.rgen = rand.New(rand.NewSource(seed))
	} else {
		m.rgen = conf.RGen
	}

	m.MultivTConf = conf
	m.d, _ = distmv.NewStudentsT(make([]float64, conf.Dim), m.sigma, conf.Nu, m.rgen)
	return m
}

// Sample from a (uncorrelated) multivariate t distribution
func (m MultivT) Sample(mu core.Elemt) core.Elemt {
	var mu_ = mu.([]float64)
	var dim = len(mu_)
	var res = make([]float64, dim)
	var s = m.d.Rand(make([]float64, dim))
	for i := range s {
		res[i] = mu_[i] + s[i]
	}
	return res
}

// Density of a (uncorrelated) multivariate t distribution
func (m MultivT) Pdf(mu, x core.Elemt) float64 {
	var mu_ = mu.([]float64)
	var x_ = x.([]float64)
	var dif = make([]float64, len(mu_))
	for i := range mu_ {
		dif[i] = mu_[i] - x_[i]
	}
	return m.d.Prob(dif)
}
