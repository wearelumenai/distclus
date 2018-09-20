package zetest

import (
	"distclus/core"
	"distclus/kmeans"
	"distclus/real"
	"math"
	"reflect"
	"testing"
	"time"
)

var TestVectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

// Algorithm must be configured with GivenInitializer with 3 centers and 0 iteration
func DoTestInitialization(t *testing.T, algo core.OnlineClust) {
	var actual = PushAndRunSync(algo)
	var expected = TestVectors[:3]
	AssertCentroids(t, expected, actual)
	algo.Close()
}

// Algorithm must be configured with GivenInitializer with 3 centers
func DoTestRunSyncGiven(t *testing.T, algo core.OnlineClust) {
	var clust = PushAndRunSync(algo)
	var actual = clust.AssignAll(TestVectors, real.RealSpace{})

	var expected = [][]int{{0, 3, 4}, {1, 5}, {2, 6, 7}}
	AssertAssignation(t, expected, actual)

	algo.Close()
}

// Algorithm must be configured with KMeansPP with 3 centers
func DoTestRunSyncKMeansPP(t *testing.T, algo core.OnlineClust) {
	var clust = PushAndRunSync(algo)
	var actual = clust.AssignAll(TestVectors,real.RealSpace{})

	var expected = make([][]int, 3)
	_, i0, _ := algo.Predict(TestVectors[0], false)
	_, i1, _ := algo.Predict(TestVectors[1], false)
	_, i2, _ := algo.Predict(TestVectors[2], false)
	expected[i0] = []int{0, 3, 4}
	expected[i1] = []int{1, 5}
	expected[i2] = []int{2, 6, 7}

	AssertAssignation(t, expected, actual)

	algo.Close()
}

// Algorithm must be configured with 3 centers
func DoTestRunSyncCentroids(t *testing.T, km *kmeans.KMeans) {
	c0, _, _ := km.Predict(TestVectors[0], false)
	c1, _, _ := km.Predict(TestVectors[1], false)
	c2, _, _ := km.Predict(TestVectors[2], false)
	var actual = core.Clust{c0, c1, c2}
	var expected = core.Clust{
		[]float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3},
		[]float64{-17. / 2, -20.5 / 2, -15. / 2, -16.5 / 2, -16.5 / 2},
		[]float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3},
	}

	AssertCentroids(t, expected, actual)
}

// Algorithm must be configured with 3 centers
func DoTestRunAsync(t *testing.T, algo core.OnlineClust) {
	RunAsyncAndPush(algo)

	time.Sleep(700 * time.Millisecond)
	var obs = []float64{-9, -10, -8.3, -8, -7.5}
	var c, _, _ = algo.Predict(obs, true)

	time.Sleep(1000 * time.Millisecond)
	algo.Close()

	var cn, _, _ = algo.Predict(obs, false)
	AssertNotEqual(t, c, cn)

	var c0, _, _ = algo.Predict(TestVectors[1], false)
	AssertEqual(t, c0, cn)
}

func DoTestRunAsyncCentroids(t *testing.T, km *kmeans.KMeans) {
	c0, _, _ := km.Predict(TestVectors[0], false)
	c1, _, _ := km.Predict(TestVectors[1], false)
	c2, _, _ := km.Predict(TestVectors[2], false)
	var actual = core.Clust{c0, c1, c2}
	var expected = core.Clust{
		[]float64{23.4 / 3, 20. / 3, 23. / 3, 29.5 / 3, 30. / 3},
		[]float64{-26. / 3, -30.5 / 3, -23.3 / 3, -24.5 / 3, -24. / 3},
		[]float64{134. / 3, 133.6 / 3, 133.2 / 3, 120.4 / 3, 135.2 / 3},
	}

	AssertCentroids(t, expected, actual)
}

func DoTestWorkflow(t *testing.T, algo core.OnlineClust) {
	DoTestBeforeRun(algo, t)

	algo.Run(true)
	time.Sleep(300 * time.Millisecond)
	DoTestAfterRun(algo, t)

	algo.Close()
	DoTestAfterClose(algo, t)
}

func DoTestAfterClose(algo core.OnlineClust, t *testing.T) {
	var err error
	err = algo.Push(TestVectors[5])
	AssertError(t, err)

	_, _, err = algo.Predict(TestVectors[5], true)
	AssertError(t, err)

	_, _, err = algo.Predict(TestVectors[5], false)
	AssertNoError(t, err)
}

func DoTestAfterRun(algo core.OnlineClust, t *testing.T) {
	var err error
	err = algo.Push(TestVectors[3])
	AssertNoError(t, err)

	_, _, err = algo.Predict(TestVectors[4], true)
	AssertNoError(t, err)
}

func DoTestBeforeRun(algo core.OnlineClust, t *testing.T) {
	var err error
	algo.Push(TestVectors[0])
	algo.Push(TestVectors[1])
	err = algo.Push(TestVectors[2])
	AssertNoError(t, err)

	_, err = algo.Centroids()
	AssertError(t, err)

	_, _, err = algo.Predict(TestVectors[3], false)
	AssertError(t, err)
}

func DoTestEmpty(t *testing.T, builder func(core.Initializer) core.OnlineClust) {
	var init = core.Clust{
		[]float64{0, 0, 0, 0, 0},
		[]float64{1000, 1000, 1000, 1000, 1000},
	}
	var algorithm = builder(init.Initializer)

	PushAndRunAsync(algorithm)
	time.Sleep(300 * time.Millisecond)

	var clust, _ = algorithm.Centroids()

	if !reflect.DeepEqual(clust[1], init[1]) {
		t.Error("Expected empty cluster")
	}
}

func PushAndRunAsync(algorithm core.OnlineClust) {
	for _, elemt := range TestVectors {
		algorithm.Push(elemt)
	}
	algorithm.Run(true)
}

func RunAsyncAndPush(algo core.OnlineClust) {
	algo.Run(true)
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}
}

func PushAndRunSync(algo core.OnlineClust) core.Clust {
	for _, elemt := range TestVectors {
		algo.Push(elemt)
	}
	algo.Run(false)
	var clust, _ = algo.Centroids()
	return clust
}

func AssertCentroids(t *testing.T, expected core.Clust, actual core.Clust) {
	if len(actual) != len(expected) {
		t.Error("Expected ", len(expected), "centroids got", len(actual))
		return
	}

	for i := 0; i < len(actual); i++ {
		AssertAlmostEqual(t, expected[i].([]float64), actual[i].([]float64))
	}
}

func AssertAssignation(t *testing.T, expected [][]int, actual [][]int) {
	if !reflect.DeepEqual(actual, expected) {
		t.Error("Expected", expected, "got", actual)
	}
}

func AssertEqual(t *testing.T, expected core.Elemt, actual core.Elemt) {
	if !reflect.DeepEqual(expected, actual) {
		t.Error("Expected same elements")
	}
}
func AssertNotEqual(t *testing.T, unexpected core.Elemt, actual core.Elemt) {
	if reflect.DeepEqual(unexpected, actual) {
		t.Error("Expected different elements")
	}
}

func AssertAlmostEqual(t *testing.T, expected []float64, actual []float64) {
	if len(expected) != len(actual) {
		t.Error("Expected", len(expected), "got", len(actual))
	}

	for i := 0; i < len(expected); i++ {
		if math.Abs(expected[i]-actual[i]) > 1e-6 {
			t.Error("Expected", expected[i], "got", actual[i])
		}
	}
}

func AssertNoError(t *testing.T, err error) {
	if err != nil {
		t.Error("Expected no workflow error")
	}
}

func AssertError(t *testing.T, err error) {
	if err == nil {
		t.Error("Expected no workflow error")
	}
}

func AssertPanic(t *testing.T) {
	if x := recover(); x == nil {
		t.Error("Expected error")
	}
}
