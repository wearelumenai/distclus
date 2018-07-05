package par

import (
	"testing"
	"distclus/core"
	"distclus/algo"
	"reflect"
	"golang.org/x/exp/rand"
	"time"
	"math"
)

var mcmcConf = algo.MCMCConf{
	Dim:      5, FrameSize: 8, B: 100, Amp: 1,
	Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: core.RealSpace{},
}

var distrib = algo.NewMultivT(algo.MultivTConf{mcmcConf})

func TestParMCMCSupport_ParLoss(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)

	var clust, _ = mcmc.Centroids()
	var l1 = mcmc.Loss(mcmc, clust)
	var l2 = clust.Loss(mcmc.Data, mcmc.Space, mcmc.Norm)

	if math.Abs(l1-l2)>1e-6 {
		t.Error("Expected", l2, "got", l1)
	}
}

func TestMCMC_ParRun(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 100
	conf.InitK = 1
	var seed = uint64(1872365454256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)
	var clust, _ = mcmc.Centroids()

	for i := 0; i < len(clust); i++ {
		if reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected center change got", clust[i])
		}
	}

	mcmc.Close()

	if mcmc.AcceptRatio() == 0 {
		t.Error("Expected positive accept ratio")
	}
}

func TestMCMC_ParPredict(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)

	for i, elemt := range data {
		var c, idx, _ = mcmc.Predict(elemt, false)

		if i == 0 || i == 3 || i == 4 {
			if idx != 0 || !reflect.DeepEqual(c, data[0]) {
				t.Error("Expected center 0")
			}
		}

		if i == 1 || i == 5 {
			if idx != 1 || !reflect.DeepEqual(c, data[1]) {
				t.Error("Expected center 1")
			}
		}

		if i == 2 || i == 6 || i == 7 {
			if idx != 2 || !reflect.DeepEqual(c, data[2]) {
				t.Error("Expected center 2")
			}
		}
	}

	mcmc.Close()
}

func TestMCMC_ParPredict2(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232548913256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = NewMCMC(conf, distrib, algo.KmeansPPInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)
	var clust, _ = mcmc.Centroids()

	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = mcmc.Predict(data[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}
	}

	for i := 3; i < len(data); i++ {
		var c, idx, _ = mcmc.Predict(data[i], false)

		if i == 3 || i == 4 {
			if idx != iclust[0] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 0")
			}
		}

		if i == 5 {
			if idx != iclust[1] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 1")
			}
		}

		if i == 6 || i == 7 {
			if idx != iclust[2] || !reflect.DeepEqual(c, clust[idx]) {
				t.Error("Expected center 2")
			}
		}
	}

	mcmc.Close()
}

func TestMCMC_ParAsync(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var mcmc = NewMCMC(conf, distrib, algo.GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(true)

	time.Sleep(700 * time.Millisecond)

	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, ix, _ = mcmc.Predict(obs, true)

	time.Sleep(1000 * time.Millisecond)
	var clust, _ = mcmc.Centroids()

	if reflect.DeepEqual(clust[ix], c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = mcmc.Predict(data[1], false)
	var _, ix5, _ = mcmc.Predict(data[5], false)

	if ix != ix5 || ix != ix1 {
		t.Error("Expected same center")
	}

	mcmc.Close()
}
