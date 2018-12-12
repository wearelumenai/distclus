package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"testing"

	"golang.org/x/exp/rand"
)

var conf = core.Conf{
	ImplConf: mcmc.Conf{
		InitK:     3,
		FrameSize: 8,
		RGen:      rand.New(rand.NewSource(6305689164243)),
		B:         100, Amp: 1,
		Norm: 2, Nu: 3, McmcIter: 20,
		InitIter: 1,
	},
	SpaceConf: nil,
}
var data = []core.Elemt{
	[]float64{1., 3.4, 5.4},
	[]float64{10., 9.2, 12.3},
	[]float64{-4.3, -1.2, -3.},
}
var initializer = kmeans.GivenInitializer

func Test_NewSeqAlgo(t *testing.T) {
	var algo = mcmc.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
		distrib,
	)
	algo.AcceptRatio()
}

func Test_NewParAlgo(t *testing.T) {
	mcmcConf := conf.ImplConf.(mcmc.Conf)
	mcmcConf.Par = true
	mcmc.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
		distrib,
	)
}

func Test_AlgoAcceptRatio(t *testing.T) {
	var algo = mcmc.NewAlgo(conf, real.Space{}, data, initializer)

	algo.Run(false)

	var acceptRatio = algo.AcceptRatio()
	var implAcceptRatio = algo.Impl().(*mcmc.Impl).AcceptRatio()

	if acceptRatio != implAcceptRatio {
		t.Error("Wrong routing of acceptratio method", acceptRatio, implAcceptRatio)
	}
}

func Test_Reset(t *testing.T) {
	algo := mcmc.NewAlgo(
		conf,
		real.Space{},
		data,
		initializer,
		distrib,
	)

	test.DoTestReset(t, &algo, core.Conf{ImplConf: mcmc.Conf{InitK: 1, MaxK: 1}, SpaceConf: nil})
}
