package core_test

import (
	"distclus/core"
	"distclus/figures"
	"distclus/internal/test"
	"errors"
	"testing"
)

type mockConf struct {
	core.Conf
}

type mockImpl struct {
	init         int
	runningcount int
	stoppedcount int
	oc           bool
	clust        core.Clust
	iter         int
}

var errInit = errors.New("init")
var errIter = errors.New("iter")

var errChan = make(chan error)
var statusChan = make(chan core.ClustStatus)

var notifier = func(status core.ClustStatus, err error) {
	if err != nil {
		errChan <- err
	}
	statusChan <- status
}

func (impl *mockImpl) Init(conf core.ImplConf, space core.Space, clust core.Clust) (centroids core.Clust, err error) {
	centroids = impl.clust
	impl.init++
	impl.iter = 0
	if len(centroids) == 1 {
		err = errInit
	}
	return
}

func (impl *mockImpl) Iterate(conf core.ImplConf, space core.Space, centroids core.Clust) (clust core.Clust, runtimeFigures figures.RuntimeFigures, err error) {
	impl.iter++
	clust = impl.clust
	runtimeFigures = figures.RuntimeFigures{"iter": figures.Value(impl.iter)}
	if len(centroids) == 2 {
		err = errIter
	}
	return
}

func (impl *mockImpl) Push(elemt core.Elemt, running bool) (err error) {
	if running {
		impl.runningcount++
	} else {
		impl.stoppedcount++
	}
	return
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

func newAlgo(t *testing.T, iter int, size int) (algo core.Algo) {
	algo = *core.NewAlgo(
		&mockConf{
			core.Conf{
				StatusNotifier: notifier,
				Iter:           iter,
			},
		},
		&mockImpl{
			clust: make(core.Clust, size),
		},
		mockSpace{},
	)

	// initialization
	// var conf = algo.Conf.(mockConf)
	var impl = algo.Impl().(*mockImpl)

	if impl.init > 0 {
		t.Error("initialized before starting")
	}
	if impl.iter > 0 {
		t.Error("running before starting")
	}
	if impl.runningcount != 0 {
		t.Error("count before starting")
	}

	return
}

func TestErrorAtInitialization(t *testing.T) {
	var algo = newAlgo(t, 1, 10)
	var impl = algo.Impl().(*mockImpl)
	impl.clust = impl.clust[0:1]

	err := algo.Batch()

	if err == nil {
		t.Error("no error in wrong cluster")
	}
}

func TestOCInitError(t *testing.T) {
	algo := newAlgo(t, 1, 1)

	err := algo.Play()

	if err != errInit {
		t.Error("error during oc initialization", err)
	}
}

func TestOCIterError(t *testing.T) {
	algo := newAlgo(t, 1, 2)

	err := algo.Play()

	if err != nil {
		t.Error("error while initializing")
	}

	_ = algo.Stop()
}

func TestOCPause(t *testing.T) {
	algo := newAlgo(t, 1, 10)

	err := algo.Play()

	if err != nil {
		t.Error("error while initializing", err)
	}

	err = algo.Pause()

	if err != nil {
		t.Error("error while pausing", err)
	}

	if algo.Status() != core.Idle {
		t.Error("Idle status expected. found", algo.Status())
	}

	err = algo.Play()

	if err != nil {
		t.Error("error while playing", err)
	}

	if algo.Status() != core.Running && algo.Status() != core.Sleeping {
		t.Error("wrong status. Running or sleeping expected", algo.Status())
	}

	_ = algo.Stop()
}

func Test_Conf(t *testing.T) {
	var algo = newAlgo(t, 1, 10)

	var conf = algo.Conf()

	if &conf == nil {
		t.Error("conf is nil")
	}
}

func Test_Predict(t *testing.T) {
	var algo = newAlgo(t, 1, 10)

	_, _, err := algo.Predict(nil)

	if err == nil {
		t.Error("initialized before running")
	}

	err = algo.Batch()

	if err != nil {
		t.Error("error while running", err)
	}

	pred, label, err := algo.Predict(nil)

	if err != nil {
		t.Error("Error while predict")
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl().(*mockImpl).runningcount != 0 {
		t.Error("element has been pushed")
	}

	pred, label, err = algo.Predict(nil)

	if err != nil {
		t.Error("Error while predict", err)
	}

	err = algo.Push(nil)

	if err != nil {
		t.Error("Error while pushing", err)
	}
	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl().(*mockImpl).runningcount != 0 && algo.Impl().(*mockImpl).stoppedcount != 1 {
		t.Error("element has not been pushed", algo.Impl().(*mockImpl).runningcount)
	}

	err = algo.Stop()

	if err != core.ErrNotRunning {
		t.Error("error while stopping the algorithm", err)
	}

	_, _, err = algo.Predict(nil)
	if err == nil {
		err = algo.Push(nil)
	}

	if err != nil {
		t.Error("error after close and prediction", err)
	}
}

func Test_Scenario_Batch(t *testing.T) {
	var algo = newAlgo(t, 1, 10)
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
	if impl.init > 0 {
		t.Error("initialized before starting")
	}
	if impl.iter > 0 {
		t.Error("running before starting")
	}
	if impl.stoppedcount != 1 {
		t.Error("count before starting")
	}

	err = algo.Batch()

	if err != nil {
		t.Error("Error while stopping")
	}

	if algo.Status() != core.Ready {
		t.Error("status should be Ready", algo.Status())
	}

	if impl.init == 0 {
		t.Error("not initialized")
	}
	var iter = impl.iter
	if iter == 0 {
		t.Error("not running")
	}

	err = algo.Stop()

	if algo.Status() != core.Ready {
		t.Error("status should be ready")
	}

	if impl.iter != iter {
		t.Error("running", impl.iter)
	}

	if err != core.ErrNotRunning {
		t.Error("Error while stopping", err)
	}
}

func Test_Scenario_Play(t *testing.T) {
	var algo = newAlgo(t, 0, 10)
	if algo.Status() != core.Created {
		t.Error("status should be Created")
	}

	var figures0, err0 = algo.RuntimeFigures()
	var iter0, ok0 = figures0[figures.Iterations]

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
	if impl.init > 0 {
		t.Error("initialized before starting")
	}
	if impl.iter > 0 {
		t.Error("running before starting")
	}
	if impl.stoppedcount != 1 {
		t.Error("count before starting")
	}

	_ = algo.Play()

	if algo.Status() != core.Running && algo.Status() != core.Sleeping {
		t.Error("status should be Running", algo.Status())
	}

	var figures1, err1 = algo.RuntimeFigures()
	var iter1, ok1 = figures1[figures.Iterations]

	test.AssertNoError(t, err1)
	test.AssertTrue(t, ok1)
	test.AssertTrue(t, iter1 > 0)

	if impl.init == 0 {
		t.Error("not initialized")
	}
	var iter = impl.iter
	if impl.iter == 0 {
		t.Error("not running")
	}

	if impl.iter < iter {
		t.Error("not running")
	}

	err = algo.Stop()

	if err != nil {
		t.Error("Error while stopping")
	}

	if algo.Status() != core.Ready {
		t.Error("status should be ready")
	}
	iter = impl.iter
	if impl.iter != iter {
		t.Error("running")
	}

	var figures2, err2 = algo.RuntimeFigures()
	var iter2, ok2 = figures2[figures.Iterations]

	test.AssertNoError(t, err2)
	test.AssertTrue(t, ok2)
	test.AssertTrue(t, iter2 >= iter1)
}
