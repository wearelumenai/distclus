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

func NewMultivariateT(dim int, tau, nu float64, src rand.Source) (MultivariateT, bool) {
	var m MultivariateT
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
	return m, ok
}

func (m* MultivariateT) Sample(mu []float64) []float64 {
	var dim = len(mu)
	var res = make([]float64, dim)
	var s = m.d.Rand(make([]float64, dim))
	for i := range s {
		res[i] = mu[i] + s[i]
	}
	return res
}

func (m* MultivariateT) Pdf(mu, x []float64) float64 {
	var dif = make([]float64, len(mu))
	for i := range mu {
		dif[i] = mu[i] - x[i]
	}
	return m.d.Prob(dif)
}