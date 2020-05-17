package mcmc_test

import (
	"math"
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"
)

func Test_Distrib(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 3,
	}
	var initializer = kmeans.GivenInitializer
	var distrib = mcmc.NewMultivT(tConf)
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)
	test.DoTestInitGiven(t, algo)
}

func newAlgo(t *testing.T, conf core.CtrlConf, size int) (algo *core.Algo) {
	var tConf = mcmc.MultivTConf{
		Dim: 3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var implConf = mcmc.Conf{InitK: 3, CtrlConf: conf}
	var initializer = kmeans.GivenInitializer
	var clust = make(core.Clust, size)
	for i := range clust {
		clust[i] = []float64{0, 1, 2}
	}
	return mcmc.NewAlgo(implConf, space, clust, initializer, distrib)
}

func Test_Scenario_Batch(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	test.DoTestScenarioBatch(t, algo)
}

func Test_scenario_infinite(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{}, 10)

	test.DoTestScenarioInfinite(t, algo)
}

func Test_scenario_finite(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	test.DoTestScenarioFinite(t, algo)
}

func Test_Scenario_Play(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 20}, 10)

	test.DoTestScenarioPlay(t, algo)
}

func Test_Timeout(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Timeout: 1, Iter: math.MaxInt64}, 10)

	test.DoTestTimeout(t, algo)
}

func Test_Freq(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{IterFreq: 10}, 10)

	test.DoTestFreq(t, algo)
}

func Test_IterToRun(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{}, 10)

	test.DoTestIterToRun(t, algo)
}
