package kmeans_test

import (
	"math"
	"testing"

	"github.com/wearelumenai/distclus/v0/core"
	"github.com/wearelumenai/distclus/v0/internal/test"
	"github.com/wearelumenai/distclus/v0/kmeans"
)

func newAlgo(t *testing.T, conf core.Conf, size int) (algo *core.Algo) {
	var implConf = kmeans.Conf{K: 3, Conf: conf}
	var initializer = kmeans.GivenInitializer
	var clust = make(core.Clust, size)
	for i := range clust {
		clust[i] = []float64{0, 1, 2}
	}
	return kmeans.NewAlgo(implConf, space, clust, initializer)
}

func Test_Scenario_Batch(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 1}, 10)

	test.DoTestScenarioBatch(t, algo)
}

func Test_scenario_infinite(t *testing.T) {
	var algo = newAlgo(t, core.Conf{}, 10)

	test.DoTestScenarioInfinite(t, algo)
}

func Test_scenario_finite(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 1000}, 10)

	test.DoTestScenarioFinite(t, algo)
}

func Test_Scenario_Play(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 20}, 10)

	test.DoTestScenarioPlay(t, algo)
}

func Test_Timeout(t *testing.T) {
	algo := newAlgo(t, core.Conf{Timeout: 1, Iter: math.MaxInt64}, 10)

	test.DoTestTimeout(t, algo)
}

func Test_Freq(t *testing.T) {
	algo := newAlgo(t, core.Conf{IterFreq: 1}, 10)

	test.DoTestFreq(t, algo)
}

func Test_IterToRun(t *testing.T) {
	algo := newAlgo(t, core.Conf{}, 10)

	test.DoTestIterToRun(t, algo)
}
