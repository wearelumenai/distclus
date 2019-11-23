package core_test

import (
	"distclus/core"
	"distclus/figures"
	"errors"
	"math"
	"testing"
	"time"
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
	if impl.iter%2 == 0 {
		runtimeFigures = figures.RuntimeFigures{"iter": figures.Value(impl.iter)}
	}
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
	impl.clust = append(impl.clust, elemt)
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

func newAlgo(t *testing.T, conf core.Conf, size int) (algo core.Algo) {
	algo = *core.NewAlgo(
		&mockConf{
			Conf: conf,
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
	var algo = newAlgo(t, core.Conf{Iter: 1}, 10)
	var impl = algo.Impl().(*mockImpl)
	impl.clust = impl.clust[0:1]

	err := algo.Batch()

	if err == nil {
		t.Error("no error in wrong cluster")
	}
}

func TestInitError(t *testing.T) {
	algo := newAlgo(t, core.Conf{Iter: 1}, 1)
	err := algo.Play()

	if err != errInit {
		t.Error("error during oc initialization", err)
	}
}

func TestIterError(t *testing.T) {
	algo := newAlgo(t, core.Conf{Iter: 1}, 2)
	err := algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Wait()

	if err != errIter {
		t.Error("Iter error expected", err)
	}

	err = algo.Wait()

	if err != errIter {
		t.Error("Iter error expected", err)
	}

	err = algo.Batch()

	if err != errIter {
		t.Error("Iter error expected", err)
	}
}

func TestPause(t *testing.T) {
	algo := newAlgo(t, core.Conf{Iter: 1}, 10)

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
	var algo = newAlgo(t, core.Conf{Iter: 1}, 10)

	var conf = algo.Conf()

	if &conf == nil {
		t.Error("conf is nil")
	}
}

func Test_Init(t *testing.T) {
	var algo = newAlgo(t, core.Conf{}, 10)

	var err = algo.Init()

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Init()

	if err != core.ErrAlreadyCreated {
		t.Error("already created expected", err)
	}
}

func Test_Predict(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 1}, 10)

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

func Test_infinite_Batch(t *testing.T) {
	var algo = newAlgo(t, core.Conf{}, 10)

	var err = algo.Batch()

	if err != core.ErrInfiniteIterations {
		t.Error("Infinite iterations expected")
	}

}

func Test_Scenario_Batch(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 1}, 10)
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

	if algo.Status() != core.Succeed {
		t.Error("status should be Succeed", algo.Status())
	}

	if impl.init == 0 {
		t.Error("not initialized")
	}
	var iter = impl.iter
	if iter == 0 {
		t.Error("not running")
	}

	err = algo.Stop()

	if algo.Status() != core.Succeed {
		t.Error("status should be ready")
	}

	if impl.iter != iter {
		t.Error("running", impl.iter)
	}

	if err != core.ErrNotRunning {
		t.Error("Error while stopping", err)
	}
}

func Test_scenario_infinite(t *testing.T) {
	var algo = newAlgo(t, core.Conf{}, 10)

	if algo.Status() != core.Created {
		t.Error("created expected", algo.Status())
	}

	var err = algo.Wait()

	if err != core.ErrNotStarted {
		t.Error("not started expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Running {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Pause()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Idle {
		t.Error("Idle expected", algo.Status())
	}

	err = algo.Pause()

	if err != core.ErrIdle {
		t.Error("EddIdle expected", err)
	}

	err = algo.Wait()

	if err != core.ErrIdle {
		t.Error("idle error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Running() {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Wait()

	if err != core.ErrNeverSleeping {
		t.Error("never sleeping expected", err)
	}
	if algo.Status() != core.Running {
		t.Error("Sleeping expected", algo.Status())
	}

	err = algo.Play()

	if err != core.ErrRunning {
		t.Error("running expected")
	}

	err = algo.Stop()

	if err != nil {
		t.Error("No error expected")
	}
	if algo.Status() != core.Succeed {
		t.Error("succeed expected", algo.Status())
	}

	err = algo.Play()
	if err != nil {
		t.Error("No error expected")
	}

	if algo.Status() != core.Running {
		t.Error("running expected", algo.Status())
	}

	err = algo.Pause()

	if err != nil {
		t.Error("No error expected")
	}

	if algo.Status() != core.Idle {
		t.Error("idle expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("No error expected")
	}

	if algo.Status() != core.Succeed {
		t.Error("Succeed expected", algo.Status())
	}
}

func Test_scenario_finite(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 10}, 10)

	if algo.Status() != core.Created {
		t.Error("created expected", algo.Status())
	}

	err := algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Running() {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Pause()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Idle {
		t.Error("Idle expected", algo.Status())
	}

	err = algo.Pause()

	if err != core.ErrIdle {
		t.Error("EddIdle expected", err)
	}

	err = algo.Wait()

	if err != core.ErrIdle {
		t.Error("idle error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Running() {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("never sleeping expected", err)
	}
	if algo.Status() != core.Sleeping {
		t.Error("Sleeping expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Sleeping {
		t.Error("Sleeping expected", algo.Status())
	}

	err = algo.Push(nil)

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Running() {
		t.Error("running expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Sleeping {
		t.Error("Sleeping expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Succeed {
		t.Error("Succeed expected", algo.Status())
	}
}

func Test_Scenario_Play(t *testing.T) {
	var algo = newAlgo(t, core.Conf{Iter: 20}, 10)
	if algo.Status() != core.Created {
		t.Error("status should be Created")
	}

	var figures0, err0 = algo.RuntimeFigures()
	var iter0, ok0 = figures0[figures.Iterations]

	if err0 != core.ErrNotStarted {
		t.Error("not started error", err0)
	}
	if ok0 {
		t.Error("no value expected", iter0)
	}

	centroids, err := algo.Centroids()

	if centroids != nil {
		t.Error("nil centroids expected", centroids)
	}
	if err != core.ErrNotStarted {
		t.Error("running expected", err)
	}

	err = algo.Push(nil)

	if err != nil {
		t.Error("No error expected", err)
	}

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
	if impl.runningcount != 0 {
		t.Error("running count before starting", impl.runningcount)
	}
	if impl.stoppedcount != 1 {
		t.Error("stopped count before starting", impl.stoppedcount)
	}

	err = algo.Play()

	if err != nil {
		t.Error("No error expected", err)
	}

	if !algo.Running() {
		t.Error("status should be Running", algo.Status())
	}

	algo.Wait()

	var figures1, err1 = algo.RuntimeFigures()
	var iter1 = figures1[figures.Iterations]

	if err1 != nil {
		t.Error("no error expected", err1)
	}
	if int(iter1) != algo.Conf().AlgoConf().Iter {
		t.Errorf("%d iterations expected. %d", algo.Conf().AlgoConf().Iter, int(iter1))
	}

	err = algo.Play()

	if err != nil {
		t.Error("No error expected", err)
	}

	if !algo.Running() {
		t.Error("status should be Running", algo.Status())
	}

	algo.Wait()

	var figures2, err2 = algo.RuntimeFigures()
	var iter2 = figures2[figures.Iterations]

	if err2 != nil {
		t.Error("no error expected", err2)
	}
	if int(iter2) != (algo.Conf().AlgoConf().Iter * 2) {
		t.Errorf("%d iterations expected. %d", algo.Conf().AlgoConf().Iter*2, int(iter1))
	}

	err = algo.Stop()

	if err != nil {
		t.Error("Error while stopping", err)
	}

	if algo.Status() != core.Succeed {
		t.Error("status should be succeed", algo.Status())
	}

	var figures3, err3 = algo.RuntimeFigures()
	var iter3 = figures3[figures.Iterations]

	if err3 != nil {
		t.Error("No error expected", err3)
	}
	if iter3 != iter2 {
		t.Errorf("%d iterations expected. %d", int(iter2), int(iter3))
	}
}

func Test_Timeout(t *testing.T) {
	algo := newAlgo(t, core.Conf{Timeout: 0.0001, Iter: math.MaxInt64}, 10)

	err := algo.Batch()

	if err != core.ErrTimeOut {
		t.Error("timeout expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Wait()

	if err != core.ErrTimeOut {
		t.Error("timeout expected", err)
	}

}
func Test_Freq(t *testing.T) {
	algo := newAlgo(t, core.Conf{IterFreq: 1}, 10)

	algo.Play()
	time.Sleep(1)
	algo.Pause()

	var runtimeFigures, _ = algo.RuntimeFigures()

	if runtimeFigures[figures.Iterations] > 1 {
		t.Error("1 iteration expected", runtimeFigures[figures.Iterations])
	}
}

func Test_StatusNotifier(t *testing.T) {
	var statusChan = make(chan core.ClustStatus, 10)
	var errorChan = make(chan error, 10)

	var statusNotifier = func(status core.ClustStatus, err error) {
		statusChan <- status
		errorChan <- err
	}

	algo := newAlgo(t, core.Conf{Iter: 1, StatusNotifier: statusNotifier}, 2)

	algo.Batch()

	var status = []core.ClustStatus{
		core.Ready, core.Running, core.Stopping, core.Failed,
	}
	var errors = []error{nil, nil, errIter, errIter}
	for _, s := range status {
		var ss, ok = <-statusChan
		if !ok {
			t.Error("status expected")
		} else {
			if ss != s {
				t.Errorf("%d expected. %d", s, ss)
			}
		}
	}

	if len(statusChan) != 0 {
		t.Error("no more status expected", len(statusChan))
	}

	for _, e := range errors {
		var ee, ok = <-errorChan
		if !ok {
			t.Error("error expected")
		} else {
			if ee != e {
				t.Errorf("%d expected. %d", e, ee)
			}
		}
	}

	if len(errorChan) != 0 {
		t.Error("no more error expected", len(errorChan))
	}

}
