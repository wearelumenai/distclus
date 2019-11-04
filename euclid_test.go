package distclus

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/kmeans"
	"distclus/mcmc"
	"testing"

	"gonum.org/v1/gonum/stat/distuv"
)

func BenchmarkVectors(b *testing.B) {
	for n := 0; n < b.N; n++ {
		centroids, err := runVectors()
		if err != nil {
			b.Fatal(err)
		}
		b.Logf("run #%v: %v centers", n, len(centroids))
	}
}

func TestVectors(t *testing.T) {
	centroids, err := runVectors()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v centers", len(centroids))
}

func runVectors() ([]core.Elemt, error) {
	var vectors = makeVectors()
	var mcmcConf, space = getVectorsConf()
	var algo = getVectorsAlgo(space, mcmcConf)
	return runVectorsAlgo(algo, vectors)
}

func getVectorsConf() (mcmc.Conf, euclid.Space) {
	var mcmcConf = mcmc.Conf{
		InitK: 2,
		Amp:   .01,
		B:     200,
		Conf: core.Conf{
			Iter: 50,
		},
	}
	var space = euclid.NewSpace(euclid.Conf{})
	return mcmcConf, space
}

func getVectorsAlgo(space euclid.Space, mcmcConf mcmc.Conf) *core.Algo {
	var initializer = kmeans.PPInitializer
	var distrib = mcmc.NewDirac()
	return mcmc.NewAlgo(mcmcConf, space, []core.Elemt{}, initializer, distrib)
}

func runVectorsAlgo(algo *core.Algo, series [][]float64) ([]core.Elemt, error) {
	for s := range series {
		if err := algo.Push(series[s]); err != nil {
			return nil, err
		}
	}

	if err := algo.Run(); err != nil {
		return nil, err
	}

	return algo.Centroids()
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
	var vectors = make([][]float64, 100000)
	for n := 0; n < 100000; n++ {
		var i = int(mix.Rand())
		vectors[n] = []float64{components[i].Rand()}
	}
	return vectors
}
