package distclus

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
	"gonum.org/v1/gonum/stat/distuv"
	"log"
	"testing"
)

func BenchmarkVectors(b *testing.B) {
	for n := 0; n < b.N; n++ {
		err := runVectors(n)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func TestVectors(t *testing.T) {
	err := runVectors(0)
	if err != nil {
		t.Fatal(err)
	}
}

func runVectors(n int) error {
	var vectors = makeVectors()
	var mcmcConf, space = getVectorsConf()
	var algo = getVectorsAlgo(space, mcmcConf)
	var err = runVectorsAlgo(algo, vectors, n)
	return err
}

func getVectorsConf() (mcmc.Conf, euclid.Space) {
	var mcmcConf = mcmc.Conf{
		InitK:    2,
		Amp:      .01,
		B:        200,
		McmcIter: 50,
	}
	var space = euclid.NewSpace(euclid.Conf{})
	return mcmcConf, space
}

func getVectorsAlgo(space euclid.Space, mcmcConf mcmc.Conf) *core.Algo {
	var initializer = kmeans.PPInitializer
	var distrib = mcmc.NewDirac()
	return mcmc.NewAlgo(mcmcConf, space, []core.Elemt{}, initializer, distrib)
}

func runVectorsAlgo(algo *core.Algo, series [][]float64, n int) error {
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

func makeVectors() [][]float64 {
	var components = []distuv.Normal{
		{Mu: 10.0, Sigma: 1.0},
		{Mu: 20.0, Sigma: 1.0},
		{Mu: 30.0, Sigma: 1.0},
		{Mu: 40.0, Sigma: 1.0},
		{Mu: 50.0, Sigma: 1.0},
	}
	var mix = distuv.NewCategorical([]float64{.2, .2, .2, .2, .2}, nil)
	var vectors = make([][]float64, 10000)
	for n := 0; n < 10000; n++ {
		var i = int(mix.Rand())
		vectors[n] = []float64{components[i].Rand()}
	}
	return vectors
}