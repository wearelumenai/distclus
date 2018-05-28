package clustering_go

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type MultivariateT struct {
	d       *distmv.StudentsT
	sigma   mat.Symmetric
	dim     int
	tau, nu float64
}

func NewMultivariateT() MultivariateT {
	return MultivariateT{}
}

func (m* MultivariateT) Init(dim int, tau, nu float64, src rand.Source) bool {
	var sigma = make([]float64, dim*dim)
	var nextX int
	var nextY int
	for i := range sigma {
		var x = i/dim
		var y = i%dim
		if x == nextX && y == nextY{
			sigma[i] = tau
			nextX += 1
			nextY += 1
		}
	}
	m.sigma = mat.NewSymDense(dim, sigma)
	m.dim = dim
	m.tau = tau
	m.nu = nu
	var d, ok = distmv.NewStudentsT(make([]float64, dim), m.sigma, nu, src)
	m.d = d
	return ok
}

func (m* MultivariateT) Sample(mu Elemt) Elemt {
	var mu_ = mu.([]float64)
	var dim = len(mu_)
	var res = make([]float64, dim)
	var s = m.d.Rand(make([]float64, dim))
	for i := range s {
		res[i] = mu_[i] + s[i]
	}
	return res
}

func (m* MultivariateT) Pdf(mu, x Elemt) float64 {
	var mu_ = mu.([]float64)
	var x_ = x.([]float64)
	var dif = make([]float64, len(mu_))
	for i := range mu_ {
		dif[i] = mu_[i] - x_[i]
	}
	return m.d.Prob(dif)
}