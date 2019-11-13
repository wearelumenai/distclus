package distclus

import (
	"distclus/core"
	"distclus/dtw"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
	"testing"

	"gonum.org/v1/gonum/stat/distuv"
)

func BenchmarkSeries(b *testing.B) {
	for n := 0; n < b.N; n++ {
		centroids, err := runSeries()
		if err != nil {
			b.Fatal(err)
		}
		b.Logf("run #%v: %v centers", n, len(centroids))
	}
}

func TestSeries(t *testing.T) {
	centroids, err := runSeries()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v centers", len(centroids))
}

func runSeries() ([]core.Elemt, error) {
	var series = makeSeries()
	var mcmcConf, space = getSeriesConf()
	var algo = getSeriesAlgo(space, mcmcConf)
	return runSeriesAlgo(algo, series)
}

func getSeriesConf() (mcmc.Conf, dtw.Space) {
	var mcmcConf = mcmc.Conf{
		InitK: 2,
		Amp:   .001,
		B:     200,
		Par:   true,
		Conf: core.Conf{
			Iter: 50,
		},
	}
	var space = dtw.NewSpace(dtw.Conf{
		InnerSpace: euclid.NewSpace(euclid.Conf{}),
		Window:     4,
	})
	return mcmcConf, space
}

func getSeriesAlgo(space dtw.Space, mcmcConf mcmc.Conf) *core.Algo {
	var initializer = kmeans.PPInitializer
	var distrib = mcmc.NewDirac()
	return mcmc.NewAlgo(mcmcConf, space, []core.Elemt{}, initializer, distrib)
}

func runSeriesAlgo(algo *core.Algo, series [][][]float64) ([]core.Elemt, error) {
	for s := range series {
		if err := algo.Push(series[s]); err != nil {
			return nil, err
		}
	}

	if err := algo.Run(false); err != nil {
		return nil, err
	}

	return algo.Centroids()
}

func makeSeries() [][][]float64 {
	var components = []distuv.Normal{
		{Mu: 10.0, Sigma: 1.0},
		{Mu: 20.0, Sigma: 1.0},
		{Mu: 30.0, Sigma: 1.0},
		{Mu: 40.0, Sigma: 1.0},
		{Mu: 50.0, Sigma: 1.0},
	}
	var mix = distuv.NewCategorical([]float64{.2, .2, .2, .2, .2}, nil)
	var series = make([][][]float64, 300)
	for n := 0; n < 300; n++ {
		series[n] = make([][]float64, 100)
		var i = int(mix.Rand())
		for t := 0; t < 100; t++ {
			series[n][t] = []float64{components[i].Rand()}
		}
	}
	return series
}
