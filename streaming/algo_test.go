package streaming_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/streaming"
	"distclus/vectors"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
	"testing"
)

func Test_StreamingAlgo(t *testing.T) {
	var algo = streaming.NewAlgo(streaming.Conf{}, vectors.Space{}, []core.Elemt{})
	var distr = mix()
	_ = algo.Push(distr())
	_ = algo.Run(true)
	for i := 0; i < 75; i++ {
		_ = algo.Push(distr())
	}
	_ = algo.Close()
	var clusters, _ = algo.Centroids()
	if c := len(clusters); c < 3 {
		t.Error("3 or more clusters expected got", c)
	}
	if len(clusters) > 6 {
		t.Error("less than 6 clusters expected")
	}
}

func Test_AlgoErr(t *testing.T) {
	defer test.AssertPanic(t)
	var _ = streaming.NewAlgo(streaming.Conf{BufferSize: 1}, vectors.Space{}, []core.Elemt{[]float64{1.}, []float64{1.}})
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
