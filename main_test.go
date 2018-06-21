package main

import (
	"testing"
	"time"
	"distclus/core"
	"distclus/algo"
	"golang.org/x/exp/rand"
	"distclus/algo/par"
)

func BenchmarkRun(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b1(b.Log)
	}
}

func b1(log func(args ...interface{})) {
	var data []core.Elemt
	var distrib algo.MCMCDistrib
	var initializer = algo.RandInitializer
	var seed = int(time.Now().UTC().Unix())
	var mcmcConf = algo.MCMCConf{
	}
	mcmcConf.Space = core.RealSpace{}
	in := "cas.csv"
	data, mcmcConf.Dim = parseFloatCsv(&in)
	mcmcConf.FrameSize = 15000
	mcmcConf.RGen = rand.New(rand.NewSource(uint64(seed)))
	mcmcConf.McmcIter = 200
	mcmcConf.B = 1
	mcmcConf.Amp = 20000
	mcmcConf.R = .1
	mcmcConf.InitIter = 0
	mcmcConf.InitK = 1
	mcmcConf.Norm = 2
	mcmcConf.Nu = 3
	distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})
	var mcmc = par.NewMCMC(mcmcConf, distrib, initializer, nil)

	mcmc.Run(true)

	go func() {
		for _, elt := range data {
			mcmc.Push(elt)
		}
	}()

	time.Sleep(15 * time.Second)

	mcmc.Close()

	var centers, _ = mcmc.Centroids()
	var labels = make([]int, len(centers))
	for i := range data {
		var _, l, _ = centers.Assign(data[i], mcmcConf.Space)
		labels[l] += 1
	}

	log(len(mcmc.Data))
	log(labels)
	log(mcmc.Loss(mcmc, centers))
}
