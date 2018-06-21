package algo

import (
	"distclus/core"
	"testing"
	"reflect"
	"time"
	"golang.org/x/exp/rand"
	"unsafe"
)

var mcmcConf = MCMCConf{
	Dim:      5, FrameSize: 8, B: 100, Amp: 1,
	Norm:     2, Nu: 3, InitK: 3, McmcIter: 20,
	InitIter: 0, Space: core.RealSpace{},
}

var distrib = NewMultivT(MultivTConf{mcmcConf})

func TestMCMC_Centroids(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(false)
	var clust, _ = mcmc.Centroids()

	for i := 0; i < conf.InitK; i++ {
		if !reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected", data[i], "got", clust[i])
		}
	}

	mcmc.Close()
}

func TestMCMC_getCenters(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

	for _, elemt := range data {
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

	var ptr = func (elemt core.Elemt) unsafe.Pointer {
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

func TestMCMC_Run(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 100
	conf.InitK = 1
	var seed = uint64(1872365454256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

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

func TestMCMC_Predict(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

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

func TestMCMC_Predict2(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	var seed = uint64(187232548913256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var mcmc = NewMCMC(conf, distrib, KmeansPPInitializer, nil)

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

func TestMCMC_Close(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var mcmc = NewMCMC(conf, distrib, GivenInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	if mcmc.status != Created {
		t.Error("Expected status", Created, "got", mcmc.status)
	}

	mcmc.Run(true)

	if mcmc.status != Running {
		t.Error("Expected status", Running, "got", mcmc.status)
	}

	mcmc.Close()

	if mcmc.status != Closed {
		t.Error("Expected status", Closed, "got", mcmc.status)
	}
}

func TestMCMC_Async(t *testing.T) {
	var conf = mcmcConf
	conf.ProbaK = []float64{1, 8, 1}
	conf.McmcIter = 1 << 30
	var mcmc = NewMCMC(conf, distrib, KmeansPPInitializer, nil)

	for _, elemt := range data {
		mcmc.Push(elemt)
	}

	mcmc.Run(true)

	time.Sleep(100 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, ix, _ = mcmc.Predict(obs, true)

	time.Sleep(600 * time.Millisecond)
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

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var mcmc = NewMCMC(conf, distrib, KmeansPPInitializer, nil)

	var err error

	err = mcmc.Push(data[0])
	err = mcmc.Push(data[1])
	err = mcmc.Push(data[2])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, err = mcmc.Centroids()

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = mcmc.Predict(data[3], false)

	if err == nil {
		t.Error("Expected workflow error")
	}

	mcmc.Run(true)

	err = mcmc.Push(data[3])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, _, err = mcmc.Predict(data[4], true)

	if err != nil {
		t.Error("Expected no workflow error")
	}

	mcmc.Close()

	err = mcmc.Push(data[5])

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = mcmc.Predict(data[5], true)

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = mcmc.Predict(data[5], false)

	if err != nil {
		t.Error("Expected no workflow error")
	}
}

func TestMCMC_Conf(t *testing.T) {
	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	func() {
		defer testPanic()
		var conf = mcmcConf
		conf.McmcIter = -10
		NewMCMC(conf, distrib, KmeansPPInitializer, nil)
	}()

	func() {
		defer testPanic()
		var conf = mcmcConf
		conf.InitK = -3
		NewMCMC(conf, distrib, KmeansPPInitializer, nil)
	}()
}