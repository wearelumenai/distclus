package streaming_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/figures"
	"distclus/internal/test"
	"distclus/streaming"
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
)

func Test_(t *testing.T) {
	const n = 1000
	var dataset = [n]core.Elemt{}
	for i := 0; i < n; i++ {
		dataset[i] = []float64{4, 1, 2}
	}

	var algo = streaming.NewAlgo(
		streaming.Conf{
			Conf:       core.Conf{Iter: 0},
			Mu:         0.5,
			Sigma:      0.1,
			OutRatio:   2,
			OutAfter:   7,
			BufferSize: 1000,
		},
		euclid.Space{},
		[]core.Elemt{},
	)

	algo.Push(dataset[0])

	algo.Play(0, 0)

	for _, data := range dataset[1:] {
		algo.Push(data)
	}

	algo.Centroids()

	// monitoring during 1 seconds since algo is running in a asynchronous mode
	for k := 0; k < 10; k++ {
		algo.Centroids()
	}
	algo.Close()
}

func Test_Iter(t *testing.T) {
	var size = 20
	var algo = newAlgo(t, core.Conf{}, size)

	algo.Batch(size-1, 0) // first data is processed at initialization

	var rf, _ = algo.RuntimeFigures()
	var iterations = rf[figures.Iterations]

	if iterations != float64(size-1) {
		t.Errorf("Wrong iterations number %f. %d Expected.", iterations, size-1)
	}
}

func Test_Async(t *testing.T) {
	var algo = streaming.NewAlgo(streaming.Conf{Conf: core.Conf{Iter: 0}}, euclid.Space{}, []core.Elemt{})
	var distr = mix()
	err := algo.Push(distr())
	if err != nil {
		t.Error("No error expected.", err)
	}
	err = algo.Play(0, 0)
	if err != nil {
		t.Error("No error expected.", err)
	}
	for i := 0; i < 999; i++ {
		_ = algo.Push(distr())
	}
	err = algo.Stop()
	if err != nil {
		t.Error("No error expected", err)
	}
	clusters, err := algo.Centroids()
	if err != nil {
		t.Error("No error expected", err)
	}
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c, algo.Status())
	}
	if len(clusters) > 9 {
		t.Error("less than 9 clusters expected")
	}
}

func Test_Sync(t *testing.T) {
	var algo = streaming.NewAlgo(streaming.Conf{Conf: core.Conf{Iter: 20}}, euclid.Space{}, []core.Elemt{})
	var distr = mix()
	var err error
	for i := 0; i < 1000; i++ {
		err = algo.Push(distr())
		if i > 99 {
			if err == nil {
				t.Error("Buffer is not full")
			}
		} else if err != nil {
			t.Error("No error expected", err)
		}
	}
	err = algo.Batch(0, 0)
	if err != nil {
		t.Error("No error expected", err)
	}
	clusters, err := algo.Centroids()
	if err != nil {
		t.Error("No error expected", err)
	}
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c)
	}
	if len(clusters) > 9 {
		t.Error("less than 9 clusters expected")
	}
}

func Test_AlgoErr(t *testing.T) {
	defer test.AssertPanic(t)
	var _ = streaming.NewAlgo(streaming.Conf{BufferSize: 1}, euclid.Space{}, []core.Elemt{[]float64{1.}, []float64{1.}})
}

func mix() func() []float64 {
	var norm1, _ = distmv.NewNormal([]float64{1., 1.}, mat.NewDiagDense(2, []float64{2., 2.}), nil)
	var norm2, _ = distmv.NewNormal([]float64{-23., 9.}, mat.NewDiagDense(2, []float64{4., 2.}), nil)
	var norm3, _ = distmv.NewNormal([]float64{-12., -25.}, mat.NewDiagDense(2, []float64{2., 4.}), nil)
	var p = distuv.Uniform{
		Min: 0.,
		Max: 1.,
	}
	return func() []float64 {
		switch a := p.Rand(); {
		case a < .2:
			return norm1.Rand(nil)
		case a < .5:
			return norm2.Rand(nil)
		default:
			return norm3.Rand(nil)
		}
	}
}

func Test_AlgoPush(t *testing.T) {
	var data = mix()
	var algo = streaming.NewAlgo(streaming.Conf{BufferSize: 5, Conf: core.Conf{Iter: 0}}, euclid.Space{}, []core.Elemt{})
	_ = algo.Push(data())
	_ = algo.Play(0, 0)
	var d = make([][]float64, 10000)
	for i := range d {
		d[i] = data()
	}
	for i := range d {
		_ = algo.Push(d[i])
	}
	_ = algo.Stop()
	var rfigures, _ = algo.RuntimeFigures()
	if rfigures[figures.MaxDistance] < 10 {
		t.Error("max distance should be grater than 1", rfigures[figures.MaxDistance])
	}
}

func newAlgo(t *testing.T, conf core.Conf, size int) (algo *core.Algo) {
	var implConf = streaming.Conf{Conf: conf, BufferSize: size}
	var clust = make(core.Clust, size)
	for i := range clust {
		clust[i] = []float64{0, 1, 2}
	}
	return streaming.NewAlgo(implConf, euclid.Space{}, clust)
}

func Test_Scenario_Batch(t *testing.T) {
	// var algo = newAlgo(t, core.Conf{Iter: 1}, 10)

	// test.DoTestScenarioBatch(t, algo)
}

func Test_scenario_infinite(t *testing.T) {
	var algo = newAlgo(t, core.Conf{}, 10)

	test.DoTestScenarioInfinite(t, algo)
}

func Test_scenario_finite(t *testing.T) {
	// var algo = newAlgo(t, core.Conf{}, 1000)

	// test.DoTestScenarioFinite(t, algo)
}

func Test_Scenario_Play(t *testing.T) {
	// var algo = newAlgo(t, core.Conf{Iter: 20}, 10)

	// test.DoTestScenarioPlay(t, algo)
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
	// algo := newAlgo(t, core.Conf{}, 10)

	// test.DoTestIterToRun(t, algo)
}
