package mcmc

import (
	"distclus/core"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
)

// Configuration for multivariateT distribution
type MultivTConf struct {
	MCMCConf
}

// Real(float64[]) distribution wrapping StudentsT of Gonum
type MultivT struct {
	MultivTConf
	normal *distmv.Normal
	chi2   *distuv.ChiSquared
	k      float64
	i      float64
}

// Constructor for multivT distribution
func NewMultivT(conf MultivTConf) MultivT {
	mu := make([]float64, conf.Dim)
	var s = make([]float64, conf.Dim)

	for i := 0; i < conf.Dim; i++ {
		s[i] = 1
		mu[i] = 0.
	}
	var sigma = mat.NewDiagonal(conf.Dim, s)

	var m = MultivT{}
	m.MultivTConf = conf
	m.normal, _ = distmv.NewNormal(mu, sigma, m.RGen)
	m.chi2 = &distuv.ChiSquared{K: conf.Nu, Src: m.RGen}
	m.i = (float64(conf.Dim) + conf.Nu) / 2.
	m.k = math.Log(math.Gamma(m.i) / math.Gamma(float64(conf.Nu)/2.))

	return m
}

// Sample from a (uncorrelated) multivariate t distribution
func (m MultivT) Sample(mu core.Elemt) core.Elemt {
	var g = math.Sqrt(m.chi2.K / m.chi2.Rand())
	var z = m.normal.Rand(nil)
	var cmu = mu.([]float64)
	var tau = m.Tau()

	for i := range z {
		z[i] = cmu[i] + z[i]*g*tau
	}

	return z
}

// Density of a (uncorrelated) multivariate t distribution
func (m MultivT) Pdf(mu, x core.Elemt) float64 {
	var cmu = mu.([]float64)
	var cx = x.([]float64)
	var nk = 0.

	for i := range cmu {
		f := cmu[i] - cx[i]
		nk += f * f
	}

	var d = m.Nu * m.Tau()
	var k = m.k - float64(m.Dim)/2.*math.Log(math.Pi*d)
	return k + (-m.i * math.Log(1.+nk/d))
}
