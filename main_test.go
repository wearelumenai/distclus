package main

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/vectors"
	"testing"
	"time"

	"golang.org/x/exp/rand"
)

func BenchmarkRun(b *testing.B) {
	for n := 0; n < b.N; n++ {
		b1(b.Log)
	}
}

func b1(log func(args ...interface{})) {
	var data []core.Elemt
	var distrib mcmc.Distrib
	var initializer = kmeans.RandInitializer
	var seed = int(time.Now().UTC().Unix())
	var conf = mcmc.Conf{}
	var space = vectors.Space{}
	in := "bf.csv"
	data, conf.Dim = parseFloatCsv(&in)
	conf.FrameSize = 0
	conf.RGen = rand.New(rand.NewSource(uint64(seed)))
	conf.McmcIter = 2
	conf.B = 1
	conf.Amp = 0.5
	conf.R = 1
	conf.InitIter = 0
	conf.InitK = 2
	conf.MaxK = 50
	conf.Norm = 2
	conf.Nu = 3
	distrib = mcmc.NewMultivT(mcmc.MultivTConf{Conf: conf})
	var impl = mcmc.NewSeqImpl(conf, initializer, nil, func(mcmc.Conf) mcmc.Distrib { return distrib })
	var algo = core.NewAlgo(core.Conf{ImplConf: conf}, &impl, space)

	for _, elt := range data {
		_ = algo.Push(elt)
	}

	_ = algo.Run(false)

	_ = algo.Close()

	var centers, _ = algo.Centroids()
	var labels = make([]int, len(centers))
	for i := range data {
		var _, l, _ = centers.Assign(data[i], space)
		labels[l]++
	}

	log(labels)
}
