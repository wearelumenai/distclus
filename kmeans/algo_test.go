package kmeans_test

import (
	"math"
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/kmeans"
)

func newAlgo(t *testing.T, conf core.CtrlConf, size int) (algo *core.Algo) {
	var implConf = kmeans.Conf{K: 3, CtrlConf: conf}
	var initializer = kmeans.GivenInitializer
	var clust = make(core.Clust, size)
	for i := range clust {
		clust[i] = []float64{0, 1, 2}
	}
	return kmeans.NewAlgo(implConf, space, clust, initializer)
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
	var algo = newAlgo(t, core.CtrlConf{Iter: 1000}, 10)

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
