package mcmc_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
	"github.com/wearelumenai/distclus/kmeans"
	"github.com/wearelumenai/distclus/mcmc"

	"gonum.org/v1/gonum/mat"

	"golang.org/x/exp/rand"
)

func Test_ParPredict_Given(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100,
		Amp:       1,
		Norm:      2,
		Par:       true,
		FrameSize: 8,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestInitGiven(t, algo)
}

func Test_ParPredictPP(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		ProbaK:    []float64{1, 8, 1},
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100,
		Amp:       0.1,
		Norm:      2,
		Par:       true,
		FrameSize: 8,
		CtrlConf: core.CtrlConf{
			Iter: 20,
		},
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.PPInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestRunSyncPP(t, algo)
}

func Test_ParRunAsync(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		ProbaK:    []float64{1, 8, 1},
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100,
		Amp:       0.1,
		Par:       true,
		FrameSize: 8,
		CtrlConf:  core.CtrlConf{Iter: 1000},
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.DoTestRunAsync(t, algo)
	test.DoTestRunAsyncPush(t, algo)
}

func TestParStrategy_Loss(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:     3,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100,
		Amp:       1,
		Norm:      2,
		Par:       true,
		FrameSize: 8,
	}
	var tConf = mcmc.MultivTConf{
		Dim: 5,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.GivenInitializer
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)

	test.PushAndInit(algo)

	var strategy = mcmc.ParStrategy{}
	buffer := core.NewDataBuffer(test.Vectors, implConf.FrameSize)
	strategy.Degree = runtime.NumCPU()

	var clust = algo.Centroids()
	var l1 = strategy.Loss(implConf, algo.Space(), clust, buffer.Data())
	var l2 = clust.TotalLoss(test.Vectors, algo.Space(), implConf.Norm)

	if math.Abs(l1-l2) > 1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}

func Test_Normal(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK:    3,
		RGen:     rand.New(rand.NewSource(6305689164243)),
		B:        1,
		Amp:      .05,
		Norm:     2,
		Par:      true,
		CtrlConf: core.CtrlConf{Iter: 60},
	}
	var tConf = mcmc.MultivTConf{
		Dim: 3,
		Nu:  3,
	}
	var distrib = mcmc.NewMultivT(tConf)
	var initializer = kmeans.RandInitializer
	var centroids, data = test.GenerateData(10000)
	var algo = mcmc.NewAlgo(implConf, space, data, initializer, distrib)

	_ = algo.Batch(nil, 0)
	var result = algo.Centroids()

	var _, cards = result.ParReduceLoss(data, space, implConf.Norm, runtime.NumCPU())

	var resultMean = test.Mean(result, cards)
	var dataMean = test.Mean(data, nil)

	test.AssertArrayAlmostEqual(t, dataMean, resultMean)

	var result0, _, _ = algo.Predict(centroids[0])
	AssertDistance(t, centroids[0], result0)
	var result1, _, _ = algo.Predict(centroids[1])
	AssertDistance(t, centroids[1], result1)
	var result2, _, _ = algo.Predict(centroids[2])
	AssertDistance(t, centroids[2], result2)
}

func AssertDistance(t *testing.T, center core.Elemt, actual core.Elemt) {
	var diff = mat.NewVecDense(3, nil)
	diff.SubVec(mat.NewVecDense(3, actual.([]float64)), mat.NewVecDense(3, center.([]float64)))
	var dist = mat.Norm(diff, 2)
	if dist > 1 {
		t.Error("to far from distribution center")
	}
}
