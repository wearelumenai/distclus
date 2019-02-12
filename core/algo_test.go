package core_test

import (
	"distclus/core"
	"distclus/internal/test"
	"errors"
	"math"
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
	error       string
	iter        int
}

func (impl *mockImpl) Init(conf core.ImplConf, space core.Space) (centroids core.Clust, err error) {
	centroids = impl.clust
	impl.initialized = true
	impl.iter = 0
	if len(centroids) == 1 {
		err = errors.New("clustering")
	}
	return
}

func (impl *mockImpl) Run(conf core.ImplConf, space core.Space, centroids core.Clust, notifier core.Notifier, closing <-chan bool, closed chan<- bool) (err error) {
	var mockConf = conf.(mockConf)
	if impl.error != "" {
		err = errors.New(impl.error)
	}
	impl.running = true
	for iter, loop := 0, true; iter < mockConf.Iter && loop; iter++ {
		impl.iter++
		select {
		case <-closing:
			loop = false
			impl.running = false
			closed <- true
			time.Sleep(300 * time.Millisecond)
		default:
			notifier(impl.clust, map[string]float64{"iterations": float64(impl.iter)})
		}
	}
	return
}

func (impl *mockImpl) errorResult() error {
	if impl.error == "" {
		return nil
	} else {
		return errors.New(impl.error)
	}
}

func (impl *mockImpl) Push(elemt core.Elemt) error {
	impl.count++
	return impl.errorResult()
}

func (impl *mockImpl) SetAsync() error {
	impl.async = true
	return impl.errorResult()
}

type mockSpace struct {
	combine int
	copy    int
}

func (mockSpace) Dist(e1, e2 core.Elemt) float64 {
	return 42.
}

func (m mockSpace) Combine(e1 core.Elemt, w1 int, e2 core.Elemt, w2 int) core.Elemt {
	m.combine++
	return e1
}

func (m mockSpace) Copy(e core.Elemt) core.Elemt {
	m.copy++
	return e
}

func (m mockSpace) Dim(e []core.Elemt) int {
	return 0
}

func newAlgo(t *testing.T, iter int) (algo core.Algo) {
	algo = core.NewAlgo(
		core.Conf{
			ImplConf: mockConf{
				Iter: iter,
			},
			SpaceConf: nil,
		},
		&mockImpl{
			clust: make(core.Clust, 10),
		},
		mockSpace{},
	)

	// initialization
	// var conf = algo.Conf.(mockConf)
	var impl = algo.Impl().(*mockImpl)

	if impl.initialized {
		t.Error("initialized before starting")
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

func TestErrorAtInitialization(t *testing.T) {
	var algo = newAlgo(t, 1)
	var impl = algo.Impl().(*mockImpl)
	impl.clust = impl.clust[0:1]

	err := algo.Run(false)

	if err == nil {
		t.Error("no error in wrong cluster")
	}
}

func TestAsyncError(t *testing.T) {
	algo := newAlgo(t, 1)

	impl := algo.Impl().(*mockImpl)

	err := algo.Run(true)

	if err != nil {
		t.Error("error during async execution")
	}

	impl.error = "launch error\n"

	err = algo.Run(true)

	if err == nil {
		t.Error("no error in wrong cluster")
	}

	_ = algo.Close()
}

func Test_Conf(t *testing.T) {
	var algo = newAlgo(t, 1)

	var conf = algo.Conf()

	if &conf == nil {
		t.Error("conf is nil")
	}
}

func Test_Predict(t *testing.T) {

	var algo = newAlgo(t, 1)

	_, _, err := algo.Predict(nil)

	if err == nil {
		t.Error("initialized before running")
	}

	err = algo.Run(false)

	if err != nil {
		t.Error("error while running prediction")
	}

	pred, label, err := algo.Predict(nil)

	if err != nil {
		t.Error("Error after initialization")
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl().(*mockImpl).count != 0 {
		t.Error("element has been pushed")
	}

	pred, label, err = algo.Predict(nil)
	_ = algo.Push(nil)

	if err != nil {
		t.Error("Error after prediction")
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl().(*mockImpl).count != 1 {
		t.Error("element has not been pushed")
	}

	err = algo.Close()

	if err != nil {
		t.Error("error while closing the algorithm")
	}

	_, _, err = algo.Predict(nil)
	if err == nil {
		err = algo.Push(nil)
	}

	if err == nil {
		t.Error("error after close and prediction")
	}
}

func Test_Scenario_Sync(t *testing.T) {
	var algo = newAlgo(t, 1)
	if algo.Status() != core.Created {
		t.Error("status should be Created")
	}

	var err error
	_, err = algo.Centroids()

	if err == nil {
		t.Error("centroids exist")
	}

	err = algo.Push(nil)

	if err != nil {
		t.Error("error while pushing a nil element")
	}

	// var conf = algo.Conf.(mockConf)
	var impl = algo.Impl().(*mockImpl)
	var space = algo.Space().(mockSpace)

	if space.combine != 0 {
		t.Error("combine has been done")
	}
	if space.copy != 0 {
		t.Error("any copy has been processed")
	}
	if impl.initialized {
		t.Error("initialized before starting")
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

	_ = algo.Run(false)
	if algo.Status() != core.Ready {
		t.Error("status should be Ready")
	}

	if !impl.initialized {
		t.Error("not initialized")
	}
	if !impl.running {
		t.Error("not running")
	}

	_ = algo.Close()
	if algo.Status() != core.Closed {
		t.Error("status should be Closed")
	}

	if !impl.running {
		t.Error("running")
	}
}

func Test_Scenario_ASync(t *testing.T) {
	var algo = newAlgo(t, math.MaxInt32)
	if algo.Status() != core.Created {
		t.Error("status should be Created")
	}

	var figures0, err0 = algo.RuntimeFigures()
	var iter0, ok0 = figures0["iterations"]

	test.AssertError(t, err0)
	test.AssertFalse(t, ok0)
	test.AssertTrue(t, iter0 == 0)

	_, err := algo.Centroids()

	if err == nil {
		t.Error("centroids exist")
	}

	_ = algo.Push(nil)

	var impl = algo.Impl().(*mockImpl)
	var space = algo.Space().(mockSpace)

	if space.combine != 0 {
		t.Error("combine has been done")
	}
	if space.copy != 0 {
		t.Error("any copy has been processed")
	}
	if impl.initialized {
		t.Error("initialized before starting")
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

	_ = algo.Run(true)
	if algo.Status() != core.Running {
		t.Error("status should be Running")
	}

	time.Sleep(500 * time.Millisecond)

	var figures1, err1 = algo.RuntimeFigures()
	var iter1, ok1 = figures1["iterations"]

	test.AssertNoError(t, err1)
	test.AssertTrue(t, ok1)
	test.AssertTrue(t, iter1 > 0)

	if !impl.async {
		t.Error("not async after asynchronous execution")
	}
	if !impl.initialized {
		t.Error("not initialized")
	}
	if !impl.running {
		t.Error("not running")
	}

	_ = algo.Close()
	if algo.Status() != core.Closed {
		t.Error("status should be Closed")
	}

	if impl.running {
		t.Error("running")
	}

	var figures2, err2 = algo.RuntimeFigures()
	var iter2, ok2 = figures2["iterations"]

	test.AssertNoError(t, err2)
	test.AssertTrue(t, ok2)
	test.AssertTrue(t, iter2 > iter1)
}
