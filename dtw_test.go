package distclus

import (
	"distclus/core"
	"distclus/dtw"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
	"gonum.org/v1/gonum/stat/distuv"
	"log"
	"testing"
)

func BenchmarkSeries(b *testing.B) {
	for n := 0; n < b.N; n++ {
		err := runSeries(n)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestSeries(t *testing.T) {
	err := runSeries(0)
	if err != nil {
		t.Fatal(err)
	}
}

func runSeries(n int) error {
	var series = makeSeries()
	var mcmcConf, space = getSeriesConf()
	var algo = getSeriesAlgo(space, mcmcConf)
	var err = runSeriesAlgo(algo, series, n)
	return err
}

func getSeriesConf() (mcmc.Conf, dtw.Space) {
	var mcmcConf = mcmc.Conf{
		InitK:    2,
		Amp:      .001,
		B:        200,
		McmcIter: 50,
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

func runSeriesAlgo(algo *core.Algo, series [][][]float64, n int) error {
	for s := range series {
		if err := algo.Push(series[s]); err != nil {
			return err
		}
	}

	if err := algo.Run(false); err != nil {
		return err
	}

	var centroids, _ = algo.Centroids()
	log.Printf("run %v: %v centers", n, len(centroids))
	return nil
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
	var series = make([][][]float64, 100)
	for n := 0; n < 100; n++ {
		series[n] = make([][]float64, 100)
		var i = int(mix.Rand())
		for t := 0; t < 100; t++ {
			series[n][t] = []float64{components[i].Rand()}
		}
	}
	return series
}
