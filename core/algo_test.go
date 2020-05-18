package core_test

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/wearelumenai/distclus/core"
	"github.com/wearelumenai/distclus/internal/test"
)

type mockConf struct {
	core.CtrlConf
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
var statusChan = make(chan core.OCStatus)

var notifier = func(status core.OCStatus, err error) {
	if err != nil {
		errChan <- err
	}
	statusChan <- status
}

func (impl *mockImpl) Init(model core.OCModel) (centroids core.Clust, err error) {
	centroids = impl.clust
	impl.init++
	impl.iter = 0
	if len(centroids) == 1 {
		err = errInit
	}
	return
}

func (impl *mockImpl) Iterate(model core.OCModel) (clust core.Clust, runtimeFigures core.RuntimeFigures, err error) {
	impl.iter++
	clust = impl.clust
	if impl.iter%2 == 0 {
		runtimeFigures = core.RuntimeFigures{"iter": float64(impl.iter)}
	}
	if len(model.Centroids()) == 2 {
		err = errIter
	}
	return
}

func (impl *mockImpl) Push(elemt core.Elemt, model core.OCModel) (err error) {
	if model.Status().Running() {
		impl.runningcount++
	} else {
		impl.stoppedcount++
	}
	impl.clust = append(impl.clust, elemt)
	return
}

func (impl *mockImpl) Copy(model core.OCModel) (core.Impl, error) {
	return impl, nil
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

func newAlgo(t *testing.T, conf core.CtrlConf, size int) (algo core.Algo) {
	algo = *core.NewAlgo(
		&mockConf{
			CtrlConf: conf,
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
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)
	var impl = algo.Impl().(*mockImpl)
	impl.clust = impl.clust[0:1]

	err := algo.Batch()

	if err == nil {
		t.Error("no error in wrong cluster")
	}
}

func TestInitError(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 1}, 1)
	err := algo.Play()

	if err != errInit {
		t.Error("error during oc initialization", err)
	}
}

func TestStop(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 100000}, 3)
	err := algo.Stop()

	if err != core.ErrNotAlive {
		t.Error("no error expected", err)
	}

	if algo.Status().Value != core.Created {
		t.Error("created expected", algo.Status())
	}

	err = algo.Init()

	if err != nil {
		t.Error("No error expected", err)
	}
	if algo.Status().Value != core.Ready {
		t.Error("Ready expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status().Value != core.Finished {
		t.Error("Ready expected", algo.Status())
	}
	if algo.Status().Error != nil {
		t.Error("no error expected", algo.Status())
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status().Value != core.Running {
		t.Error("running expected", algo.Status().Value)
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status().Value != core.Finished {
		t.Error("finished expected", algo.Status().Value)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status().Value != core.Running {
		t.Error("running expected", algo.Status().Value)
	}

	err = algo.Pause()

	if err != nil {
		t.Error("not running expected", err)
	}
	if algo.Status().Value != core.Idle {
		t.Error("Idle expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status().Value != core.Finished {
		t.Error("Finished expected", algo.Status())
	}
}

func TestIterError(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 100}, 2)
	err := algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Wait(nil, 0)

	if err != errIter {
		t.Error("iter error expected", err)
	}

	err = algo.Wait(nil, 1*time.Second)

	if err != core.ErrNotRunning {
		t.Error("not running expected", err)
	}

	err = algo.Batch()

	if err != errIter {
		t.Error("Iter error expected", err)
	}
}

func TestPause(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	err := algo.Play()

	if err != nil {
		t.Error("error while initializing", err)
	}

	err = algo.Pause()

	if err != nil {
		t.Error("error while pausing", err)
	}

	if algo.Status().Value != core.Idle {
		t.Error("Idle status expected. found", algo.Status())
	}

	err = algo.Play()

	if err != nil {
		t.Error("error while playing", err)
	}

	if algo.Status().Value != core.Running && algo.Status().Value != core.Finished {
		t.Error("wrong status. Running expected", algo.Status())
	}

	_ = algo.Stop()
}

func Test_Conf(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	var conf = algo.Conf()

	if &conf == nil {
		t.Error("conf is nil")
	}
}

func Test_Init(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{}, 10)

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
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	_, _, _ = algo.Predict(nil)

	var err = algo.Play()

	if err != nil {
		t.Error("error while running", err)
	}

	pred, label, _ := algo.Predict(nil)

	if pred != nil {
		t.Error("pred has been found")
	}
	if label != 0 {
		t.Error("wrong label")
	}
	if algo.Impl().(*mockImpl).runningcount != 0 {
		t.Error("element has been pushed")
	}

	pred, label, _ = algo.Predict(nil)

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
	if algo.Impl().(*mockImpl).runningcount != 1 {
		t.Error("element has not been pushed", algo.Impl().(*mockImpl).runningcount)
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error while stopping the algorithm", err)
	}

	_, _, _ = algo.Predict(nil)

	if err == nil {
		err = algo.Push(nil)
	}

	if err != nil {
		t.Error("error after close and prediction", err)
	}

	if algo.Impl().(*mockImpl).stoppedcount != 1 {
		t.Error("element has not been pushed", algo.Impl().(*mockImpl).stoppedcount)
	}
}

func Test_infinite_Batch(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{}, 10)

	var err = algo.Batch()

	if err != core.ErrNeverFinish {
		t.Error("Infinite iterations expected", err)
	}

}

func Test_Scenario_Batch(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 1}, 10)

	test.DoTestScenarioBatch(t, &algo)
}

func Test_scenario_infinite(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{}, 10)

	test.DoTestScenarioInfinite(t, &algo)
}

func Test_scenario_finite(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{}, 10)

	test.DoTestScenarioFinite(t, &algo)
}

func Test_Scenario_Play(t *testing.T) {
	var algo = newAlgo(t, core.CtrlConf{Iter: 20}, 10)

	test.DoTestScenarioPlay(t, &algo)
}

func Test_Timeout(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Timeout: 1, Iter: math.MaxInt64}, 10)

	test.DoTestTimeout(t, &algo)
}

func Test_Freq(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{IterFreq: 10}, 10)

	test.DoTestFreq(t, &algo)
}

func Test_StatusNotifier(t *testing.T) {
	var statusChan = make(chan core.OCStatus, 10)
	var errorChan = make(chan error, 10)

	var statusNotifier = func(_ core.OnlineClust, status core.OCStatus) {
		statusChan <- status
		errorChan <- status.Error
	}

	algo := newAlgo(t, core.CtrlConf{Iter: 1, StatusNotifier: statusNotifier}, 2)

	algo.Batch()

	var status = []core.ClustStatus{
		core.Initializing, core.Ready, core.Running, core.Finished,
	}
	var errors = []error{nil, nil, nil, errIter}
	for _, s := range status {
		var ss, ok = <-statusChan
		if !ok {
			t.Error("status expected")
		} else {
			if ss.Value != s {
				t.Errorf("%v expected. %v", s, ss)
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
				t.Errorf("%v expected. %v", e, ee)
			}
		}
	}

	if len(errorChan) != 0 {
		t.Error("no more error expected", len(errorChan))
	}

}

/*
func Test_SetConf(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 1000}, 10)

	test.DoTestSetConf(t, &algo)
}
*/

/*
func Test_SetSpace(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{Iter: 1000}, 10)

	test.DoTestSetSpace(t, &algo)
}
*/

func Test_IterToRun(t *testing.T) {
	algo := newAlgo(t, core.CtrlConf{}, 10)

	test.DoTestIterToRun(t, &algo)
}
