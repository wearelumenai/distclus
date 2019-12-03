package core_test

import (
	"distclus/core"
	"distclus/euclid"
	"distclus/internal/test"
	"reflect"
	"testing"
)

var testPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

func TestClust_Assign(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = euclid.Space{}
	var c, ix, d = clust.Assign(testPoints[0], sp)

	if c.([]float64)[0] != clust[0].([]float64)[0] || ix != 0 || d != 2. {
		t.Error("Expected cluster 0 at distance 2 got", ix, d)
	}
}

func TestClust_ReduceDBA(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = euclid.Space{}
	var result, cards = clust.ReduceDBA(testPoints, sp)

	for i, e := range result {
		switch i {
		case 0:
			if e.([]float64)[0] < 0 {
				t.Error("Expected non negative elements")
			}
			if cards[i] != 9 {
				t.Error("Expected 9 got", cards[i])
			}
		case 1:
			if e.([]float64)[0] >= 0 {
				t.Error("Expected negative elements")
			}
			if cards[i] != 5 {
				t.Error("Expected 5 got", cards[i])
			}
		}
	}
}

func TestClust_MapLabel(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = euclid.Space{}
	var result, dists = clust.MapLabel(testPoints, sp)

	for i, label := range result {
		if label == 0 && testPoints[i].([]float64)[0] < 0 {
			t.Error("Expected non negative elements")
		}
		if label == 1 && testPoints[i].([]float64)[0] >= 0 {
			t.Error("Expected negative elements")
		}
		if dists[i] > 10 {
			t.Error("Expected distance to be < 10")
		}
	}
}

func TestClust_Loss(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = euclid.Space{}

	var expected = 0.
	for _, e := range testPoints {
		var d = Distance2Mean(e)
		expected += d * d
	}

	var actual = clust.TotalLoss(testPoints, sp, 2.)

	if expected != actual {
		t.Error("Expected", expected, "got", actual)
	}
}

func Distance2Mean(elemt core.Elemt) float64 {
	var f = elemt.([]float64)[0]
	if f < 0 {
		return f + 1
	} else {
		return f
	}
}

func TestDBA(t *testing.T) {
	var sp = euclid.Space{}

	var _, err = core.DBA([]core.Elemt{}, sp)

	if err == nil {
		t.Error("Expected empty error")
	}

	var average = 0.
	for i := 0; i < len(testPoints); i++ {
		average += (testPoints[i]).([]float64)[0]
	}
	average = average / float64(len(testPoints))
	AssertDBA(t, testPoints, sp, []float64{average})

	var elemts = make([]core.Elemt, len(testPoints))
	for i := range testPoints {
		e := (testPoints[i]).([]float64)[0]
		elemts[i] = []float64{e, 2 * e}
	}
	AssertDBA(t, elemts, sp, []float64{average, 2 * average})
}

func TestWeightedDBA(t *testing.T) {
	var sp = euclid.Space{}

	var average = 0.
	var n = len(testPoints)
	var weights = make([]int, n)
	var total = 0.
	for i := 0; i < n; i++ {
		weights[i] = i + 1
		average += (testPoints[i]).([]float64)[0] * float64(i+1)
		total += float64(i + 1)
	}
	average = average / total
	AssertWeightedDBA(t, testPoints, weights, sp, []float64{average})
}

func AssertDBA(t *testing.T, elemts []core.Elemt, sp euclid.Space, average []float64) {
	var dba, _ = core.DBA(elemts, sp)
	for i := range average {
		if value := dba.([]float64)[i]; value != average[i] {
			t.Error("Expected", average[i], "got", value)
		}
	}
}

func AssertWeightedDBA(t *testing.T, elemts []core.Elemt, weights []int, sp euclid.Space, average []float64) {
	var dba, _ = core.WeightedDBA(elemts, weights, sp)
	for i := range average {
		if value := dba.([]float64)[i]; value != average[i] {
			t.Error("Expected", average[i], "got", value)
		}
	}
}

func TestClust_Initializer(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = euclid.Space{}
	var c, _ = clust.Initializer(2, testPoints, sp, nil)

	if !reflect.DeepEqual(clust, c) {
		t.Error("Expected identity")
	}
}

func TestClust_Empty(t *testing.T) {
	func() {
		defer test.AssertPanic(t)
		var clust = core.Clust{}
		clust.MapLabel(testPoints, euclid.Space{})
	}()
}

func TestClust_ReduceDBA2(t *testing.T) {
	var centroids, data = test.GenerateData(10000)

	var dbas, cards = centroids.ReduceDBA(data, euclid.NewSpace())

	var dbasAverage = test.Mean(dbas, cards)
	var dataAverage = test.Mean(data, nil)

	test.AssertArrayAlmostEqual(t, dataAverage, dbasAverage)
}

func TestClust_DBAForLabels(t *testing.T) {
	var centroids, data = test.GenerateData(10000)

	var labels = make([]int, 10000)
	var space = euclid.NewSpace()
	var means, cards = centroids.ReduceDBAForLabels(data, labels, space)

	var meansAverage = test.Mean(means, cards)
	var dataAverage = test.Mean(data, nil)

	test.AssertArrayAlmostEqual(t, dataAverage, meansAverage)
}

func TestClust_LossForLabels(t *testing.T) {
	var centroids, data = test.GenerateData(10000)

	var labels = make([]int, 10000)
	var space = euclid.NewSpace()
	var loss, cards = centroids.ReduceLossForLabels(data, labels, space, 2.)

	if cards[0] != 10000 {
		t.Error("cardinality error")
	}
	var clusterLoss = centroids.TotalLoss(data, space, 2.)
	if clusterLoss >= loss[0] {
		t.Error("loss error")
	}
}
