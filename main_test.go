package main

import (
	"distclus/kmeans"
	"testing"
	"time"
	"distclus/core"
	"distclus/mcmc"
	"distclus/real"
	"golang.org/x/exp/rand"
)

func BenchmarkRun(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b1(b.Log)
	}
}

func b1(log func(args ...interface{})) {
	var data []core.Elemt
	var distrib mcmc.MCMCDistrib
	var initializer = kmeans.RandInitializer
	var seed = int(time.Now().UTC().Unix())
	var mcmcConf = mcmc.MCMCConf{
	}
	mcmcConf.Space = real.RealSpace{}
	in := "cas.csv"
	data, mcmcConf.Dim = parseFloatCsv(&in)
	mcmcConf.FrameSize = 15000
	mcmcConf.RGen = rand.New(rand.NewSource(uint64(seed)))
	mcmcConf.McmcIter = 200
	mcmcConf.B = 1
	mcmcConf.Amp = 1
	mcmcConf.R = .1
	mcmcConf.InitIter = 0
	mcmcConf.InitK = 1
	mcmcConf.Norm = 2
	mcmcConf.Nu = 3
	distrib = mcmc.NewMultivT(mcmc.MultivTConf{mcmcConf})
	var algo = mcmc.NewParMCMC(mcmcConf, distrib, initializer, nil)

	algo.Run(true)

	go func() {
		for _, elt := range data {
			algo.Push(elt)
		}
	}()

	time.Sleep(15 * time.Second)

	algo.Close()

	var centers, _ = algo.Centroids()
	var labels = make([]int, len(centers))
	for i := range data {
		var _, l, _ = centers.Assign(data[i], mcmcConf.Space)
		labels[l] += 1
	}

	log(len(algo.Data))
	log(labels)
	log(algo.Loss(centers))
}
