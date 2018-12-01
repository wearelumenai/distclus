package main

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
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
	var space = real.Space{}
	in := "cas.csv"
	data, conf.Dim = parseFloatCsv(&in)
	conf.FrameSize = 15000
	conf.RGen = rand.New(rand.NewSource(uint64(seed)))
	conf.McmcIter = 200
	conf.B = 1
	conf.Amp = 1
	conf.R = .1
	conf.InitIter = 0
	conf.InitK = 1
	conf.Norm = 2
	conf.Nu = 3
	distrib = mcmc.NewMultivT(mcmc.MultivTConf{Conf: conf})
	var impl = mcmc.NewParImpl(&conf, initializer, nil, distrib)
	var algo = core.NewAlgo(conf, &impl, space)

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
		var _, l, _ = centers.Assign(data[i], space)
		labels[l]++
	}

	log(labels)
}
