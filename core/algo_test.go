package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"testing"
	"time"
)

type mockConf struct {
	Iter int
}

type mockImpl struct {
	running     bool
	initialized bool
	count       int
	async       bool
	clust       core.Clust
}

func (impl *mockImpl) Init(conf core.Conf) bool {
	impl.initialized = true
	return impl.clust != nil
}

func (impl *mockImpl) Run(conf core.Conf, space core.Space, closing <-chan bool) {
	var mockConf = conf.(mockConf)
	impl.running = true
	for iter, loop := 0, true; iter < mockConf.Iter && loop; iter++ {
		select {

		case <-closing:
			loop = false
			impl.running = false

		default:
		}
	}
}

func (impl *mockImpl) Push(elemt core.Elemt) {
	impl.count++
}

func (impl *mockImpl) SetAsync() {
	impl.async = true
}

func (impl *mockImpl) Centroids() core.Clust {
	return impl.clust
}

type mockSpace struct {
	combine int
	copy    int
}

func (mockSpace) Dist(e1, e2 core.Elemt) float64 {
	return 42.
}

func (m mockSpace) Combine(e1 core.Elemt, w1 int, e2 core.Elemt, w2 int) {
	m.combine++
}

func (m mockSpace) Copy(e core.Elemt) core.Elemt {
	m.copy++
	return e
}

func newAlgo(t *testing.T) (algo core.Algo) {
	algo = core.NewAlgo(
		mockConf{},
		&mockImpl{
			clust: make(core.Clust, 10),
		},
		mockSpace{},
	)

	// initialization
	var conf = algo.Conf.(mockConf)
	var impl = algo.Impl.(*mockImpl)

	if impl.initialized {
		t.Error("initialized before starting")
	}
	if conf.Iter != 0 {
		t.Error("iterated before starting")
	}
	if impl.running {
		t.Error("running before starting")
	}
	if impl.count != 0 {
		t.Error("count before starting")
	}
	if impl.async {
		t.Error("async before starting")
	}

	return
}

func TestError(t *testing.T) {
	defer test.AssertPanic(t)

	var algo = newAlgo(t)
	algo.Impl.(*mockImpl).clust = nil

	algo.Run(false)
}

func Test_Predict(t *testing.T) {

	var algo = newAlgo(t)

	_, _, err := algo.Predict(nil, false)

	if err == nil {
		t.Error("initialized before running")
	}

	algo.Run(false)

	pred, label, err := algo.Predict(nil, false)

	if err != nil {
		t.Error("Error after initialization")
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl.(*mockImpl).count != 0 {
		t.Error("element has been pushed")
	}

	pred, label, err = algo.Predict(nil, true)

	if err != nil {
		t.Error("Error after prediction")
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl.(*mockImpl).count != 1 {
		t.Error("element has not been pushed")
	}

	algo.Close()

	_, _, err = algo.Predict(nil, true)

	if err == nil {
		t.Error("Missing error after close and prediction")
	}
}

func Test_Scenario_Sync(t *testing.T) {
	var algo = newAlgo(t)

	var err error
	_, err = algo.Centroids()

	if err == nil {
		t.Error("centroids exist")
	}

	algo.Push(nil)

	var conf = algo.Conf.(mockConf)
	var impl = algo.Impl.(*mockImpl)
	var space = algo.Space.(mockSpace)

	if space.combine != 0 {
		t.Error("combine has been done")
	}
	if space.copy != 0 {
		t.Error("any copy has been processed")
	}
	if impl.initialized {
		t.Error("initialized before starting")
	}
	if conf.Iter != 0 {
		t.Error("iterated before starting")
	}
	if impl.running {
		t.Error("running before starting")
	}
	if impl.count != 1 {
		t.Error("count before starting")
	}
	if impl.async {
		t.Error("async before starting")
	}

	algo.Run(false)

	if !impl.initialized {
		t.Error("not initialized")
	}
	if !impl.running {
		t.Error("not running")
	}

	algo.Close()

	if !impl.running {
		t.Error("running")
	}
}

func Test_Scenario_ASync(t *testing.T) {
	var algo = newAlgo(t)

	var err error
	_, err = algo.Centroids()

	if err == nil {
		t.Error("centroids exist")
	}

	algo.Push(nil)

	var conf = algo.Conf.(mockConf)
	var impl = algo.Impl.(*mockImpl)
	var space = algo.Space.(mockSpace)

	if space.combine != 0 {
		t.Error("combine has been done")
	}
	if space.copy != 0 {
		t.Error("any copy has been processed")
	}
	if impl.initialized {
		t.Error("initialized before starting")
	}
	if conf.Iter != 0 {
		t.Error("iterated before starting")
	}
	if impl.running {
		t.Error("running before starting")
	}
	if impl.count != 1 {
		t.Error("count before starting")
	}
	if impl.async {
		t.Error("async before starting")
	}

	algo.Run(true)

	time.Sleep(500. * time.Millisecond)

	if !impl.async {
		t.Error("not async after asynchronous execution")
	}
	if !impl.initialized {
		t.Error("not initialized")
	}
	if !impl.running {
		t.Error("not running")
	}

	impl.clust = nil

	algo.Run(true)

	if !impl.async {
		t.Error("not async after asynchronous execution")
	}
	if !impl.initialized {
		t.Error("not initialized")
	}
	if !impl.running {
		t.Error("not running")
	}

	algo.Close()

	if !impl.running {
		t.Error("running")
	}
}
