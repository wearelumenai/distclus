package algo

import (
	"distclus/core"
	"testing"
	"reflect"
	"time"
	"golang.org/x/exp/rand"
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
	var km = NewMCMC(conf, distrib, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < conf.InitK; i++ {
		if !reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected", data[i], "got", clust[i])
		}
	}

	km.Close()
}

func TestMCMC_Run(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 100
	conf.InitK = 1
	var seed = uint64(1872365454256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var km = NewMCMC(conf, distrib, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	for i := 0; i < len(clust); i++ {
		if reflect.DeepEqual(clust[i], data[i]) {
			t.Error("Expected center change got", clust[i])
		}
	}

	km.Close()
}

func TestMCMC_Predict(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 0
	var km = NewMCMC(conf, distrib, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)

	for i, elemt := range data {
		var c, idx, _ = km.Predict(elemt, false)

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

	km.Close()
}

func TestMCMC_Predict2(t *testing.T) {
	var conf = mcmcConf
	var seed = uint64(187236548914256543)
	conf.RGen = rand.New(rand.NewSource(seed))
	var km = NewMCMC(conf, distrib, KmeansPPInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(false)
	var clust, _ = km.Centroids()

	var iclust = make([]int, 3)
	for i := 0; i < 3; i++ {
		var c, ix, _ = km.Predict(data[i], false)
		iclust[i] = ix

		if !reflect.DeepEqual(c, clust[ix]) {
			t.Error("Expected center", clust[i], "got", c)
		}
	}

	for i := 3; i < len(data); i++ {
		var c, idx, _ = km.Predict(data[i], false)

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

	km.Close()
}

func TestMCMC_Close(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var km = NewMCMC(conf, distrib, GivenInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	if km.status != Created {
		t.Error("Expected status", Created, "got", km.status)
	}

	km.Run(true)

	if km.status != Running {
		t.Error("Expected status", Running, "got", km.status)
	}

	km.Close()

	if km.status != Closed {
		t.Error("Expected status", Closed, "got", km.status)
	}
}

func TestMCMC_Async(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var km = NewMCMC(conf, distrib, KmeansPPInitializer)

	for _, elemt := range data {
		km.Push(elemt)
	}

	km.Run(true)

	time.Sleep(300 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, ix, _ = km.Predict(obs, true)

	time.Sleep(600 * time.Millisecond)
	var clust, _ = km.Centroids()

	if reflect.DeepEqual(clust[ix], c) {
		t.Error("Expected center change")
	}

	var _, ix1, _ = km.Predict(data[1], false)
	var _, ix5, _ = km.Predict(data[5], false)

	if ix != ix5 || ix != ix1 {
		t.Error("Expected same center")
	}

	km.Close()
}

func TestMCMC_Workflow(t *testing.T) {
	var conf = mcmcConf
	conf.McmcIter = 1 << 30
	var km = NewMCMC(conf, distrib, KmeansPPInitializer)

	var err error

	err = km.Push(data[0])
	err = km.Push(data[1])
	err = km.Push(data[2])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, err = km.Centroids()

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[3], false)

	if err == nil {
		t.Error("Expected workflow error")
	}

	km.Run(true)

	err = km.Push(data[3])

	if err != nil {
		t.Error("Expected no workflow error")
	}

	_, _, err = km.Predict(data[4], true)

	if err != nil {
		t.Error("Expected no workflow error")
	}

	km.Close()

	err = km.Push(data[5])

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[5], true)

	if err == nil {
		t.Error("Expected workflow error")
	}

	_, _, err = km.Predict(data[5], false)

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
		NewMCMC(conf, distrib, KmeansPPInitializer)
	}()

	func() {
		defer testPanic()
		var conf = mcmcConf
		conf.InitK = -3
		NewMCMC(conf, distrib, KmeansPPInitializer)
	}()
}
