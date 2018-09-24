package mcmc_test

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/mcmc"
	"distclus/real"
	"distclus/internal/test"
	"golang.org/x/exp/rand"
	"testing"
	"reflect"
	"time"
	"unsafe"
)

func TestMCMC_getCenters(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var buffer = core.NewBuffer([]core.Elemt{}, -1)
	var seed = uint64(time.Now().UTC().Unix())
	var rgen = rand.New(rand.NewSource(seed))
	var space = real.RealSpace{}
	var store = mcmc.NewCenterStore(&buffer, space, rgen)

	for _, elemt := range test.TestVectors {
		buffer.Push(elemt)
	}

	var clust3, _ = kmeans.GivenInitializer(3, test.TestVectors, space, rgen)
	var clust = store.GetCenters(3, clust3)
	store.SetCenters(clust)

	if !reflect.DeepEqual(clust3, clust) {
		t.Error("Expected same centers")
	}

	clust = store.GetCenters(4, clust)
	store.SetCenters(clust)

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
	clust = store.GetCenters(3, clust4)
	store.SetCenters(clust)

	if !reflect.DeepEqual(clust, clust3) {
		t.Error("Expected same centers")
	}

	clust = store.GetCenters(2, clust)
	store.SetCenters(clust)

	for i := 0; i < 2; i++ {
		if !reflect.DeepEqual(clust[i], clust3[i]) && !reflect.DeepEqual(clust[i], clust3[i+1]) {
			t.Error("Expected same centers")
		}
	}

	var clust2 = clust
	clust = store.GetCenters(3, clust)
	store.SetCenters(clust)

	if !reflect.DeepEqual(clust, clust3) {
		t.Error("Expected same centers")
	}

	clust = store.GetCenters(4, clust)
	store.SetCenters(clust)

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

