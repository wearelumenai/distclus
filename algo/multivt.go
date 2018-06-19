package algo

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"distclus/core"
	"time"
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
	sigma  mat.Symmetric
	normal *distmv.Normal
	chi2   *distuv.ChiSquared
	rgen   *rand.Rand
	k      float64
	d      float64
	i      float64
}

// Constructor for multivT distribution
func NewMultivT(conf MultivTConf) MultivT {
	mu := make([]float64, conf.Dim)
	var sigma = make([]float64, conf.Dim)
	var tau = conf.Tau()

	for i := 0; i < conf.Dim; i++ {
		sigma[i] = tau
		mu[i] = 0.
	}
	var m = MultivT{}
	m.sigma = mat.NewDiagonal(conf.Dim, sigma)

	if conf.RGen == nil {
		var seed = uint64(time.Now().UTC().Unix())
		m.rgen = rand.New(rand.NewSource(seed))
	} else {
		m.rgen = conf.RGen
	}

	m.MultivTConf = conf
	m.normal, _ = distmv.NewNormal(mu, m.sigma, m.rgen)
	m.chi2 = &distuv.ChiSquared{K: conf.Nu, Src: m.rgen}
	m.i = (float64(conf.Dim) + conf.Nu) / 2.
	m.d = conf.Nu * tau
	k1 := math.Gamma(m.i)
	k2 := math.Gamma(float64(conf.Nu) / 2.)
	k3 := math.Pow(math.Pi*m.d, float64(conf.Dim)/2.)
	m.k = math.Log(k1 / k2 / k3)

	return m
}

// Sample from a (uncorrelated) multivariate t distribution
func (m MultivT) Sample(mu core.Elemt) core.Elemt {
	var g = math.Sqrt(m.chi2.K/m.chi2.Rand())
	var z = m.normal.Rand(nil)
	var cmu = mu.([]float64)
	for i := range z {
		z[i] = cmu[i] + z[i]*g
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
	return m.k + (-m.i * math.Log(1.+nk/m.d))
}
