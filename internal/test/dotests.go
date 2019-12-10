package test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/figures"
	"math"
	"reflect"
	"testing"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Vectors are values to test
var Vectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

// DoTestInitialization Algorithm must be configured with GivenInitializer with 3 centers and 0 iteration
func DoTestInitialization(t *testing.T, algo core.OnlineClust) {
	var actual = PushAndInit(algo)
	var expected = Vectors[:3]
	AssertCentroids(t, expected, actual)
}

// DoTestInitGiven Algorithm must be configured with GivenInitializer with 3 centers
func DoTestInitGiven(t *testing.T, algo core.OnlineClust) {
	var clust = PushAndInit(algo)
	var actual, _ = clust.MapLabel(Vectors, euclid.Space{})

	var expected = []int{0, 1, 2, 0, 0, 1, 2, 2}
	AssertArrayEqual(t, expected, actual)
}

// DoTestRunSyncGiven Algorithm must be configured with GivenInitializer with 3 centers
func DoTestRunSyncGiven(t *testing.T, algo core.OnlineClust) {
	var clust = PushAndRunSync(algo)
	var actual, _ = clust.MapLabel(Vectors, euclid.Space{})

	var expected = []int{0, 1, 2, 0, 0, 1, 2, 2}
	AssertArrayEqual(t, expected, actual)
}

// DoTestRunSyncPP Algorithm must be configured with PP with 3 centers
func DoTestRunSyncPP(t *testing.T, algo core.OnlineClust) {
	var clust = PushAndRunSync(algo)
	var actual, _ = clust.MapLabel(Vectors, euclid.Space{})

	_, i0, _ := algo.Predict(Vectors[0])
	_, i1, _ := algo.Predict(Vectors[1])
	_, i2, _ := algo.Predict(Vectors[2])
	var expected = []int{i0, i1, i2, i0, i0, i1, i2, i2}

	AssertArrayEqual(t, expected, actual)
}

// DoTestRunSyncCentroids Algorithm must be configured with 3 centers
func DoTestRunSyncCentroids(t *testing.T, km core.OnlineClust) {
	c0, _, _ := km.Predict(Vectors[0])
	c1, _, _ := km.Predict(Vectors[1])
	c2, _, _ := km.Predict(Vectors[2])
	var actual = core.Clust{c0, c1, c2}
	var expected = core.Clust{
		[]float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3},
		[]float64{-17. / 2, -20.5 / 2, -15. / 2, -16.5 / 2, -16.5 / 2},
		[]float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3},
	}

	AssertCentroids(t, expected, actual)
}

// DoTestRunAsync Algorithm must be configured with 3 centers
func DoTestRunAsync(t *testing.T, algo core.OnlineClust) {
	RunAsyncAndPush(algo)

	algo.Wait()

	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, _, _ = algo.Predict(obs)
	_ = algo.Push(obs)

	algo.Batch()

	var cn, _, _ = algo.Predict(obs)
	AssertNotEqual(t, c, cn)

	var c0, _, _ = algo.Predict(Vectors[1])
	AssertEqual(t, c0, cn)
}

// DoTestRunAsyncPush Algorithm must be configured with 3 centers
func DoTestRunAsyncPush(t *testing.T, algo core.OnlineClust) {
	// RunAsyncAndPush(algo)

	var figures0, err0 = algo.RuntimeFigures()
	var iter0, ok0 = figures0[figures.Iterations]

	AssertTrue(t, ok0)
	AssertNoError(t, err0)
	AssertTrue(t, iter0 > 0)

	algo.Batch()

	var figures1, err1 = algo.RuntimeFigures()
	var iter1, ok1 = figures1[figures.Iterations]

	AssertTrue(t, ok1)
	AssertNoError(t, err1)
	AssertTrue(t, iter0 < iter1)

	var centroids1, err = algo.Centroids()

	AssertNoError(t, err)

	_ = algo.Push(Vectors[0])
	_ = algo.Push(Vectors[3])
	_ = algo.Push(Vectors[5])

	algo.Batch()

	var figures2, erri2 = algo.RuntimeFigures()
	var iter2, ok2 = figures2[figures.Iterations]

	AssertTrue(t, ok2)
	AssertNoError(t, erri2)
	AssertTrue(t, iter1 < iter2)

	var centroids2, err2 = algo.Centroids()

	AssertNotEqual(t, centroids1, centroids2)

	AssertNoError(t, err2)

	_ = algo.Stop()
}

// DoTestRunAsyncCentroids test
func DoTestRunAsyncCentroids(t *testing.T, km core.OnlineClust) {
	c0, _, _ := km.Predict(Vectors[0])
	c1, _, _ := km.Predict(Vectors[1])
	c2, _, _ := km.Predict(Vectors[2])
	var actual = core.Clust{c0, c1, c2}
	var expected = core.Clust{
		[]float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3},
		[]float64{-26. / 3, -30.5 / 3, -23.3 / 3, -24.5 / 3, -24. / 3},
		[]float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3},
	}

	AssertCentroids(t, expected, actual)
}

// DoTestWorkflow test
func DoTestWorkflow(t *testing.T, algo core.OnlineClust) {
	DoTestBeforeRun(algo, t)

	_ = algo.Play()
	DoTestAfterRun(algo, t)

	_ = algo.Stop()
	DoTestAfterClose(algo, t)
}

// DoTestAfterClose test
func DoTestAfterClose(algo core.OnlineClust, t *testing.T) {
	var err error
	err = algo.Push(Vectors[5])
	AssertNoError(t, err)

	_, _, err = algo.Predict(Vectors[5])
	if err == nil {
		err = algo.Push(Vectors[5])
	}
	AssertNoError(t, err)

	_, _, err = algo.Predict(Vectors[5])
	AssertNoError(t, err)

	err = algo.Play()
	AssertNoError(t, err)
}

// DoTestAfterRun test
func DoTestAfterRun(algo core.OnlineClust, t *testing.T) {
	var err error
	err = algo.Push(Vectors[3])
	AssertNoError(t, err)

	_, _, err = algo.Predict(Vectors[4])
	_ = algo.Push(Vectors[4])

	AssertNoError(t, err)
}

// DoTestBeforeRun test
func DoTestBeforeRun(algo core.OnlineClust, t *testing.T) {
	var err error
	_ = algo.Push(Vectors[0])
	_ = algo.Push(Vectors[1])
	err = algo.Push(Vectors[2])
	AssertNoError(t, err)

	_, err = algo.Centroids()
	AssertError(t, err)

	_, _, err = algo.Predict(Vectors[3])
	AssertError(t, err)
}

// DoTestEmpty test
func DoTestEmpty(t *testing.T, builder func(core.Initializer) core.OnlineClust) {
	var init = core.Clust{
		[]float64{0, 0, 0, 0, 0},
		[]float64{1000, 1000, 1000, 1000, 1000},
	}
	var algorithm = builder(init.Initializer)

	PushAndRunAsync(algorithm)

	algorithm.Wait()

	var clust, _ = algorithm.Centroids()

	if !reflect.DeepEqual(clust[1], init[1]) {
		t.Error("Expected empty cluster")
	}
}

// PushAndInit test
func PushAndInit(algorithm core.OnlineClust) (centroids core.Clust) {
	for _, elemt := range Vectors {
		_ = algorithm.Push(elemt)
	}
	algorithm.Init()
	centroids, _ = algorithm.Centroids()
	return
}

// PushAndRunAsync test
func PushAndRunAsync(algorithm core.OnlineClust) {
	for _, elemt := range Vectors {
		_ = algorithm.Push(elemt)
	}
	_ = algorithm.Play()
}

// RunAsyncAndPush test
func RunAsyncAndPush(algo core.OnlineClust) {
	for _, elemt := range Vectors {
		_ = algo.Push(elemt)
	}
	_ = algo.Play()
}

// PushAndRunSync test
func PushAndRunSync(algo core.OnlineClust) core.Clust {
	for _, elemt := range Vectors {
		_ = algo.Push(elemt)
	}
	_ = algo.Batch()
	var clust, _ = algo.Centroids()
	return clust
}

// AssertCentroids test
func AssertCentroids(t *testing.T, expected core.Clust, actual core.Clust) {
	if len(actual) != len(expected) {
		t.Error("Expected ", len(expected), "centroids got", len(actual))
		return
	}

	for i := 0; i < len(actual); i++ {
		AssertArrayAlmostEqual(t, expected[i].([]float64), actual[i].([]float64))
	}
}

// AssertArrayEqual test
func AssertArrayEqual(t *testing.T, expected []int, actual []int) {
	if !reflect.DeepEqual(actual, expected) {
		t.Error("Expected", expected, "got", actual)
	}
}

// AssertEqual test
func AssertEqual(t *testing.T, expected core.Elemt, actual core.Elemt) {
	if !reflect.DeepEqual(expected, actual) {
		t.Error("Expected same elements")
	}
}

// AssertFalse test
func AssertFalse(t *testing.T, value bool) {
	if value {
		t.Error("False expected")
	}
}

// AssertTrue test
func AssertTrue(t *testing.T, value bool) {
	if !value {
		t.Error("True expected")
	}
}

// AssertNotEqual test
func AssertNotEqual(t *testing.T, unexpected core.Elemt, actual core.Elemt) {
	if reflect.DeepEqual(unexpected, actual) {
		t.Error("Expected different elements")
	}
}

// AssertArrayAlmostEqual test
func AssertArrayAlmostEqual(t *testing.T, expected []float64, actual []float64) {
	if len(expected) != len(actual) {
		t.Error("Expected", len(expected), "got", len(actual))
	}

	for i := 0; i < len(expected); i++ {
		AssertAlmostEqual(t, expected[i], actual[i])
	}
}

// AssertAlmostEqual test
func AssertAlmostEqual(t *testing.T, expected float64, actual float64) {
	if math.Abs(expected-actual) > 1e-6 {
		t.Error("Expected", expected, "got", actual)
	}
}

// AssertNoError test
func AssertNoError(t *testing.T, err error) {
	if err != nil {
		t.Error("Expected no workflow error")
	}
}

// AssertError test
func AssertError(t *testing.T, err error) {
	if err == nil {
		t.Error("Expected workflow error")
	}
}

// AssertPanic test
func AssertPanic(t *testing.T) {
	if x := recover(); x == nil {
		t.Error("Expected error")
	}
}

// GenerateData returns centers and sample clusters following normal distributions
func GenerateData(n int) (core.Clust, []core.Elemt) {
	var rgen = rand.New(rand.NewSource(6305689164243))
	var sigma = mat.NewDiagDense(3, []float64{1., 1., 1.})
	var centroids = core.Clust{
		[]float64{0., 0., 0.},
		[]float64{0., 15., 0.},
		[]float64{-5., -5., 5.},
	}
	var dist1, _ = distmv.NewNormal(centroids[0].([]float64), sigma, rgen)
	var dist2, _ = distmv.NewNormal(centroids[1].([]float64), sigma, rgen)
	var dist3, _ = distmv.NewNormal(centroids[2].([]float64), sigma, rgen)

	var mixed = func() distmv.Rander {
		var alpha = rgen.Float64()
		switch {
		case alpha < .2:
			return dist1
		case alpha < .5:
			return dist2
		default:
			return dist3
		}
	}

	var data = make([]core.Elemt, n)
	for i := 0; i < n; i++ {
		var x = make([]float64, 3)
		data[i] = mixed().Rand(x)
	}

	return centroids, data
}

// Mean calculates the weighted mean of the given elements
func Mean(data []core.Elemt, weights []int) []float64 {
	var s = make([]float64, len(data[0].([]float64)))
	var w = 0.
	for i := 0; i < len(data); i++ {
		if data[i] != nil {
			var weight = 1.
			if weights != nil {
				weight = float64(weights[i])
			}
			w += weight
			for j := 0; j < len(s); j++ {
				s[j] += data[i].([]float64)[j] * weight
			}
		}
	}
	for j := 0; j < len(s); j++ {
		s[j] /= w
	}
	return s
}

// DoTestScenarioBatch test batch mode
func DoTestScenarioBatch(t *testing.T, algo *core.Algo) {
	if algo.Status() != core.Created {
		t.Error("status should be Created")
	}

	var err error
	_, err = algo.Centroids()

	if err == nil {
		t.Error("centroids exist")
	}

	err = algo.Push([]float64{0, 1, 2})

	if err != nil {
		t.Error("error while pushing a nil element")
	}

	var rf, _ = algo.RuntimeFigures()
	if rf[figures.Iterations] > 0 {
		t.Error("no iterations expected")
	}

	err = algo.Batch()

	if err != nil {
		t.Error("Error while stopping", err)
	}

	if algo.Status() != core.Succeed {
		t.Error("status should be Succeed", algo.Status())
	}

	rf, _ = algo.RuntimeFigures()
	var iter = rf[figures.Iterations]
	if iter == 0 {
		t.Error("iterations expected")
	}

	err = algo.Stop()

	if algo.Status() != core.Succeed {
		t.Error("status should be ready")
	}

	rf, _ = algo.RuntimeFigures()
	if rf[figures.Iterations] != iter {
		t.Error("no iterations expected")
	}

	if err != nil {
		t.Error("no error while stopping", err)
	}
}

// DoTestScenarioInfinite test infinite case
func DoTestScenarioInfinite(t *testing.T, algo *core.Algo) { // no Iter or = 0
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

	if err != nil {
		t.Error("No error expected", err)
	}

	err = algo.Wait()

	if err != core.ErrIdle {
		t.Error("idle error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Alive() {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Wait()

	if err != core.ErrNeverEnd {
		t.Error("never end expected", err)
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
	if algo.Status() != core.Waiting {
		t.Error("Waiting expected", algo.Status())
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

	if algo.Status() != core.Waiting {
		t.Error("Waiting expected", algo.Status())
	}
}

// DoTestScenarioFinite test finite case
func DoTestScenarioFinite(t *testing.T, algo *core.Algo) { // require iter = 1000, iterPerData = 1000
	algo.Conf().AlgoConf().Iter = 1000
	algo.Conf().AlgoConf().IterPerData = 1000
	if algo.Status() != core.Created {
		t.Error("created expected", algo.Status())
	}

	err := algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Alive() {
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

	if err != nil {
		t.Error("No error expected", err)
	}

	err = algo.Wait()

	if err != core.ErrIdle {
		t.Error("idle error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Alive() {
		t.Error("Running expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("never sleeping expected", err)
	}
	if algo.Status() != core.Waiting {
		t.Error("waiting expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Waiting {
		t.Error("Waiting expected", algo.Status())
	}

	err = algo.Push([]float64{0, 1, 2})

	if err != nil {
		t.Error("no error expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	if !algo.Alive() {
		t.Error("running expected", algo.Status())
	}

	err = algo.Wait()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Waiting {
		t.Error("Waiting expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error expected", err)
	}
	if algo.Status() != core.Stopped {
		t.Error("Stopped expected", algo.Status())
	}
}

// DoTestScenarioPlay test play scenario
func DoTestScenarioPlay(t *testing.T, algo *core.Algo) { // must Iter = 20
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

	err = algo.Push([]float64{0, 1, 2})

	if err != nil {
		t.Error("No error expected", err)
	}

	var rf, _ = algo.RuntimeFigures()
	if rf[figures.Iterations] > 0 {
		t.Error("no iterations expected")
	}

	err = algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}

	if !algo.Alive() {
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

	if !algo.Alive() {
		t.Error("status should be Running", algo.Status())
	}

	algo.Wait()

	var figures2, err2 = algo.RuntimeFigures()
	var iter2 = figures2[figures.Iterations]

	if err2 != nil {
		t.Error("no error expected", err2)
	}
	var totalIter = (algo.Conf().AlgoConf().Iter + algo.Conf().AlgoConf().IterPerData)
	if int(iter2) != totalIter {
		t.Errorf("%d iterations expected. %d", totalIter, int(iter1))
	}

	err = algo.Stop()

	if err != nil {
		t.Error("no error while stopping", err)
	}

	if algo.Status() != core.Waiting {
		t.Error("status should be waiting", algo.Status())
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

// DoTestTimeout test timeout
func DoTestTimeout(t *testing.T, algo core.OnlineClust) { // Timeout 0.0001 and Iter max
	algo.Conf().AlgoConf().Timeout = 1
	algo.Conf().AlgoConf().Iter = 1000

	err := algo.Play()

	if err != nil {
		t.Error("no error expected", err)
	}
	err = algo.Wait()

	if err != core.ErrTimeOut {
		t.Error("timeout expected", err)
	}

	err = algo.Batch()

	if err != core.ErrTimeOut {
		t.Error("timeout expected", err, algo.Status())
	}

}

// DoTestFreq test frequency
func DoTestFreq(t *testing.T, algo core.OnlineClust) { // must IterFreq = 1
	algo.Play()
	time.Sleep(1)
	algo.Pause()

	var runtimeFigures, _ = algo.RuntimeFigures()

	if runtimeFigures[figures.Iterations] > 1 {
		t.Error("1 iteration expected", runtimeFigures[figures.Iterations])
	}
}

// DoTestReconfigure test reconfiguration
func DoTestReconfigure(t *testing.T, algo *core.Algo) { // must Iter = 1000

	algo.Conf().AlgoConf().Iter = 1000

	var err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != core.ErrNotStarted {
		t.Error("not started expected", err)
	}

	err = algo.Init()

	if err != nil {
		t.Error("No error expected", err)
	}

	err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != core.ErrNotStarted {
		t.Error("not started expected", err)
	}

	err = algo.Play()

	if err != nil {
		t.Error("No error expected", err)
	}

	err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != nil {
		t.Error("reconfiguration expected", err)
	}

	if algo.Status() != core.Running {
		t.Error("running expected", algo.Status())
	}

	err = algo.Pause()

	if err != nil {
		t.Error("not started expected", err)
	}

	if algo.Status() != core.Idle {
		t.Error("idle expected", algo.Status())
	}

	err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != nil {
		t.Error("reconfiguration expected", err)
	}

	if algo.Status() != core.Idle {
		t.Error("running expected", algo.Status())
	}

	err = algo.Play()

	if err != nil {
		t.Error("running expected", err)
	}

	err = algo.Wait()

	if err != nil {
		t.Error("ended expected", err)
	}

	if algo.Status() != core.Waiting {
		t.Error("Waiting expected", algo.Status())
	}

	err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != nil {
		t.Error("reconfiguration expected", err)
	}

	if algo.Status() != core.Waiting {
		t.Error("waiting expected", algo.Status())
	}

	err = algo.Stop()

	if err != nil {
		t.Error("not error expected", err)
	}

	if algo.Alive() {
		t.Error("not running expected", algo.Status())
	}

	err = algo.Reconfigure(algo.Conf(), algo.Space())

	if err != nil {
		t.Error("reconfiguration expected", err)
	}

	if algo.Status() != core.Stopped {
		t.Error("stopped expected", algo.Status())
	}
}
