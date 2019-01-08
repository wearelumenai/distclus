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
	in := "cas.csv"
	data, conf.Dim = parseFloatCsv(&in)
	conf.FrameSize = 0
	conf.RGen = rand.New(rand.NewSource(uint64(seed)))
	conf.McmcIter = 200
	conf.B = 0.05
	conf.Amp = 0.02
	conf.R = 1
	conf.InitIter = 0
	conf.InitK = 8
	conf.MaxK = 32
	conf.Norm = 2
	conf.Nu = 3
	distrib = mcmc.NewMultivT(mcmc.MultivTConf{Conf: conf})
	var impl = mcmc.NewSeqImpl(&conf, initializer, nil, distrib)
	var algo = core.NewAlgo(core.Conf{ImplConf: conf, SpaceConf: nil}, &impl, space)

	for _, elt := range data {
		algo.Push(elt)
	}

	algo.Run(false)

	algo.Close()

	var centers, _ = algo.Centroids()
	var labels = make([]int, len(centers))
	for i := range data {
		var _, l, _ = centers.Assign(data[i], space)
		labels[l]++
	}

	log(labels)
}
