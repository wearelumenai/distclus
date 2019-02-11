package mcmc_test

import (
	"distclus/core"
	"distclus/mcmc"
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
