package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"distclus/real"
	"reflect"
	"testing"
	"time"
)

var conf = core.AlgorithmConf{
	InitK: 3,
	Space: real.RealSpace{},
}

type mockAlgo struct {
	*core.AlgorithmTemplate
}

func (algo *mockAlgo) runAlgorithm(closing <-chan bool) {
	for loop := true; loop; {
		select {
		case <-closing:
			loop = false

		default:
			algo.Apply()
		}
	}
}

func newMockAlgo(init func() (core.Clust, bool)) *mockAlgo {
	var mock = mockAlgo{}
	var algoTemplateMethods = core.AlgorithmTemplateMethods {
		Initialize: init,
		Run: mock.runAlgorithm,
	}
	mock.AlgorithmTemplate = core.NewAlgo(conf, make([]core.Elemt, 0), algoTemplateMethods)
	return &mock
}

type mockInitializer struct {
	try int
}

func (*mockInitializer) NoInitialize() (core.Clust, bool) {
	return nil, false
}

func (*mockInitializer) Initialize() (core.Clust, bool) {
	var clust = test.TestVectors[:2]
	return clust, true
}

func (mock *mockInitializer) TryInitialize() (core.Clust, bool) {
	if mock.try < 2 {
		mock.try++
		return mock.NoInitialize()
	} else {
		return mock.Initialize()
	}
}

func TestAbstractAlgo_InitSync(t *testing.T) {
	defer test.AssertPanic(t)
	var algo = newMockAlgo((&mockInitializer{}).NoInitialize)
	algo.Run(false)
}

func TestAbstractAlgo_InitAsync(t *testing.T) {
	var initializer = &mockInitializer{}
	var algo = newMockAlgo(initializer.TryInitialize)
	algo.Run(true)

	var clust0, _ = algo.Centroids()
	if clust0 != nil {
		t.Error("Expected uninitialized centroids")
	}

	time.Sleep(800 * time.Millisecond)
	var clust1, _ = algo.Centroids()
	if clust1 == nil {
		t.Error("Expected initialized centroids")
	}
	if initializer.try < 2 {
		t.Error("Expected at least 2 tries")
	}
}

func TestAbstractAlgo_Push(t *testing.T) {
	var algo = newMockAlgo((&mockInitializer{}).Initialize)
	algo.Push(test.TestVectors[1])

	if len(algo.Data) != 1 {
		t.Error("Expected 1 element")
	}

	if !reflect.DeepEqual(algo.Data[0], test.TestVectors[1]) {
		t.Error("Expected .2 got", algo.Data[0])
	}
}

func TestAbstractAlgo_Predict(t *testing.T) {
	var algo = newMockAlgo((&mockInitializer{}).Initialize)
	algo.Run(true)
	time.Sleep(300 * time.Millisecond)

	var _, label, _ = algo.Predict(test.TestVectors[1], false)
	time.Sleep(300 * time.Millisecond)

	if label != 1 {
		t.Error("Expected label 1")
	}

	if len(algo.Data) != 0 {
		t.Error("Expected no element")
	}

	algo.Close()
}

func TestAbstractAlgo_PredictAndPush(t *testing.T) {
	var algo = newMockAlgo((&mockInitializer{}).Initialize)
	algo.Run(true)
	time.Sleep(300 * time.Millisecond)

	var _, label, _ = algo.Predict(test.TestVectors[1], true)
	time.Sleep(300 * time.Millisecond)

	if label != 1 {
		t.Error("Expected label 1")
	}

	if len(algo.Data) != 1 {
		t.Error("Expected 1 element")
	}

	if !reflect.DeepEqual(algo.Data[0], test.TestVectors[1]) {
		t.Error("Expected .2 got", algo.Data[0])
	}

	algo.Close()
}

func TestAbstractAlgo_Workflow(t *testing.T) {
	var algo = newMockAlgo((&mockInitializer{}).Initialize)
	test.DoTestWorkflow(t, algo)
}
