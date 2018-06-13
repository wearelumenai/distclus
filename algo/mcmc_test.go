package algo

import (
	"testing"
	"distclus/core"
	"time"
	"golang.org/x/exp/rand"
)

func TestMCMC(t *testing.T) {
	var data = make([]core.Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}
	var space = core.RealSpace{}
	var mcmcConf = MCMCConf{
		Dim:      5, FrameSize: 8, B: 100, Amp: 1,
		Norm:     2, Nu: 1, InitK: 3, McmcIter: 3,
		InitIter: 1, Initializer: RandInitializer, Space: space,
	}
	var distrib, ok = NewMultivT(MultivTConf{mcmcConf})
	if !ok {
		t.Errorf("Expected true, got false")
	}
	var mcmc = NewMCMC(mcmcConf, &distrib)
	for _, elt := range data {
		mcmc.Push(elt)
	}
	mcmc.Run()
	mcmc.Close()
	var clust, _ = mcmc.Centroids()
	res := len(clust)
	if res != 3 {
		t.Errorf("Expected 3, got %v", res)
	}
}

func TestKmeansPP(t *testing.T) {
	var data = make([]core.Elemt, 8)
	data[0] = []float64{7.2, 6, 8, 11, 10}
	data[1] = []float64{9, 8, 7, 7.5, 10}
	data[2] = []float64{7.2, 6, 8, 11, 10}
	data[3] = []float64{-9, -10, -8, -8, -7.5}
	data[4] = []float64{-8, -10.5, -7, -8.5, -9}
	data[5] = []float64{42, 41.2, 42, 40.2, 45}
	data[6] = []float64{42, 41.2, 42.2, 40.2, 45}
	data[7] = []float64{50, 51.2, 49, 40, 45.2}
	var space = core.RealSpace{}
	var src = rand.New(rand.NewSource(uint64(time.Now().UTC().Unix())))
	var clust = KmeansPPInitializer(3, data, space, src)
	res := len(clust)
	if res != 3 {
		t.Errorf("Expected 3, got %v", res)
	}
}


