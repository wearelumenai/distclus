package algo

import (
	"distclus/core"
	"distclus/real"
	"testing"
	"reflect"
	"unsafe"
)

var TestVectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

var mcmcConf = MCMCConf{
	Dim:      5, FrameSize: 8, B: 100, Amp: 1,
	Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: real.RealSpace{},
}

var distrib = NewMultivT(MultivTConf{mcmcConf})

func TestMCMC_getCenters(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

	for _, elemt := range TestVectors {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)

	var clust3, _ = mcmc.Centroids()
	var clust = mcmc.getCenters(3, clust3)

	if !reflect.DeepEqual(clust3, clust) {
		t.Error("Expected same centers")
	}

	clust = mcmc.getCenters(4, clust)

	if len(clust) != 4 {
		t.Error("Expected 4 centers")
	}

	if !reflect.DeepEqual(clust[:3], clust3) {
		t.Error("Expected same 3 fisrt centers")
	}

	for i := 0; i < 3; i++ {
		if reflect.DeepEqual(clust[3], clust3[i]) {
			t.Error("Expected different 4th center")
		}
	}

	var clust4 = clust
	clust = mcmc.getCenters(3, clust4)

	if !reflect.DeepEqual(clust, clust3) {
		t.Error("Expected same centers")
	}

	clust = mcmc.getCenters(2, clust)

	for i := 0; i < 2; i++ {
		if !reflect.DeepEqual(clust[i], clust3[i]) && !reflect.DeepEqual(clust[i], clust3[i+1]) {
			t.Error("Expected same centers")
		}
	}

	var clust2 = clust
	clust = mcmc.getCenters(3, clust)

	if !reflect.DeepEqual(clust, clust3) {
		t.Error("Expected same centers")
	}

	clust = mcmc.getCenters(4, clust)

	if !reflect.DeepEqual(clust, clust4) {
		t.Error("Expected same centers")
	}

	var ptr = func(elemt core.Elemt) unsafe.Pointer {
		var s = elemt.([]float64)
		var hdr = (*reflect.SliceHeader)(unsafe.Pointer(&s))
		return unsafe.Pointer(hdr.Data)
	}

	if ptr(clust2[0]) == ptr(clust3[0]) ||
		ptr(clust2[0]) == ptr(clust4[0]) ||
		ptr(clust3[0]) == ptr(clust4[0]) {
		t.Error("Expected copy")
	}

	if ptr(clust2[1]) == ptr(clust3[1]) ||
		ptr(clust2[1]) == ptr(clust4[1]) ||
		ptr(clust3[1]) == ptr(clust4[1]) {
		t.Error("Expected copy")
	}
	if ptr(clust3[2]) == ptr(clust4[2]) {
		t.Error("Expected copy")
	}
}

