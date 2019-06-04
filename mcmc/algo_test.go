package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"testing"
)

func Test_Distrib(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
	}
	var tConf = mcmc.MultivTConf{
		Conf: implConf,
		Dim:  3,
	}
	var initializer = kmeans.GivenInitializer
	var distrib = mcmc.NewMultivT(tConf)
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)
	test.DoTestRunSyncGiven(t, algo)
}
