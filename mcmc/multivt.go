package mcmc

import (
	"distclus/core"
	"golang.org/x/exp/rand"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
)

// MultivTConf Configuration for multivariateT distribution
type MultivTConf struct {
	Conf
	Dim int
	Nu  float64
}

// MultivT Vectors(float64[]) distribution wrapping StudentsT of Gonum
type MultivT struct {
	MultivTConf
	normal      *distmv.Normal
	chi2        *distuv.ChiSquared
	gammaFactor float64
	power       float64
}

// NewMultivT Constructor for multivT distribution
func NewMultivT(conf MultivTConf) MultivT {
	mu := make([]float64, conf.Dim)
	var s = make([]float64, conf.Dim)

	for i := 0; i < conf.Dim; i++ {
		s[i] = 1
		mu[i] = 0.
	}
	var sigma = mat.NewDiagDense(conf.Dim, s)

	var m = MultivT{}
	m.setConf(conf)
	m.normal, _ = distmv.NewNormal(mu, sigma, m.RGen)
	m.chi2 = &distuv.ChiSquared{K: m.Nu, Src: m.RGen}
	m.power = (float64(m.Dim) + m.Nu) / 2.
	m.gammaFactor = math.Log(math.Gamma(m.power) / math.Gamma(float64(m.Nu)/2.))

	return m
}

func (m *MultivT) setConf(conf MultivTConf) {
	m.MultivTConf = conf
	if conf.Nu == 0 {
		m.MultivTConf.Nu = 3
	}
	if m.RGen == nil {
		var seed = uint64(time.Now().Unix())
		m.MultivTConf.RGen = rand.New(rand.NewSource(seed))
	}
}

// Sample from a (uncorrelated) multivariate t distribution
func (m MultivT) Sample(mu core.Elemt, time int) core.Elemt {
	var fmu = mu.([]float64)
	var scale = 1 / math.Sqrt(float64(time*20))

	var chiInverse = math.Sqrt(m.chi2.K / m.chi2.Rand())
	var student = m.normal.Rand(nil)

	for i := range student {
		student[i] = fmu[i] + student[i]*chiInverse*scale
	}

	return student
}

// Pdf Density of a (uncorrelated) multivariate t distribution
func (m MultivT) Pdf(mu, x core.Elemt, time int) float64 {
	var fmu = mu.([]float64)
	var fx = x.([]float64)
	var shift = 0.

	for i := range fmu {
		f := fmu[i] - fx[i]
		shift += f * f
	}

	var scale = m.Nu / math.Sqrt(float64(time*20))
	var k = m.gammaFactor - math.Log(math.Pi*scale)*float64(m.Dim)/2.
	return k + (-m.power * math.Log(1.+shift/scale))
}
