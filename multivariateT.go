package clustering_go

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type MultivTConf struct {
	MCMCConf
}

type MultivT struct {
	c MultivTConf
	sigma   mat.Symmetric
	d       *distmv.StudentsT
	src     *rand.Rand
}

func NewMultivT(c MultivTConf) (m MultivT, ok bool) {
	var sigma = make([]float64, c.Dim*c.Dim)
	var nextX int
	var nextY int
	var tau = c.Tau()
	for i := range sigma {
		var x = i/c.Dim
		var y = i%c.Dim
		if x == nextX && y == nextY{
			sigma[i] = tau
			nextX += 1
			nextY += 1
		}
	}
	m.sigma = mat.NewSymDense(c.Dim, sigma)
	m.src = rand.New(rand.NewSource(c.Seed))
	d, ok := distmv.NewStudentsT(make([]float64, c.Dim), m.sigma, c.Nu, m.src)
	m.d = d
	return m, ok
}

func (m*MultivT) Sample(mu Elemt) Elemt {
	var mu_ = mu.([]float64)
	var dim = len(mu_)
	var res = make([]float64, dim)
	var s = m.d.Rand(make([]float64, dim))
	for i := range s {
		res[i] = mu_[i] + s[i]
	}
	return res
}

func (m*MultivT) Pdf(mu, x Elemt) float64 {
	var mu_ = mu.([]float64)
	var x_ = x.([]float64)
	var dif = make([]float64, len(mu_))
	for i := range mu_ {
		dif[i] = mu_[i] - x_[i]
	}
	return m.d.Prob(dif)
}