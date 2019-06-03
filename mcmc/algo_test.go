package mcmc_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/kmeans"
	"distclus/mcmc"
	"testing"
)

func Test_DistribBuilder(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
		Dim:   3,
	}
	var initializer = kmeans.GivenInitializer
	var distrib = func(mcmc.Conf) mcmc.Distrib {
		return mcmc.NewMultivT(mcmc.MultivTConf{implConf})
	}
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)
	test.DoTestRunSyncGiven(t, algo)
}

func Test_Distrib(t *testing.T) {
	var implConf = mcmc.Conf{
		InitK: 3,
		Dim:   3,
	}
	var initializer = kmeans.GivenInitializer
	var distrib = mcmc.NewMultivT(mcmc.MultivTConf{implConf})
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)
	test.DoTestRunSyncGiven(t, algo)
}

func Test_DistribError(t *testing.T) {
	defer test.AssertPanic(t)
	var implConf = mcmc.Conf{
		InitK: 3,
		Dim:   3,
	}
	var initializer = kmeans.GivenInitializer
	var distrib = "invalid distribution"
	var algo = mcmc.NewAlgo(implConf, space, []core.Elemt{}, initializer, distrib)
	test.DoTestRunSyncGiven(t, algo)
}
