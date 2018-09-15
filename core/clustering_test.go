package core_test

import (
	"testing"
	"reflect"
	"distclus/real"
	"distclus/core"
)

var testPoints = []core.Elemt{[]float64{2.}, []float64{4.}, []float64{1.}, []float64{8.}, []float64{-4.},
	[]float64{6.}, []float64{-10.}, []float64{0.}, []float64{-7.}, []float64{3.}, []float64{5.},
	[]float64{-5.}, []float64{-8.}, []float64{9.}}

var testVectors = []core.Elemt{
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-8, -10.5, -7, -8.5, -9},
	[]float64{42, 41.2, 42, 40.2, 45},
	[]float64{9, 8, 7, 7.5, 10},
	[]float64{7.2, 6, 8, 11, 10},
	[]float64{-9, -10, -8, -8, -7.5},
	[]float64{42, 41.2, 42.2, 40.2, 45},
	[]float64{50, 51.2, 49, 40, 45.2},
}

func TestClust_Assign(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = real.RealSpace{}
	var c, ix, d = clust.Assign(testPoints[0], sp)

	if c.([]float64)[0] != clust[0].([]float64)[0] || ix != 0 || d != 2. {
		t.Error("Expected cluster 0 at distance 2 got", ix, d)
	}
}

func TestClust_AssignDBA(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = real.RealSpace{}
	var result, cards = clust.AssignDBA(testPoints, sp)

	for i, e := range result {
		switch(i) {
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

func TestClust_AssignAll(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = real.RealSpace{}
	var result = clust.AssignAll(testPoints, sp)

	for i, c := range result {
		for _, e := range c {
			if i == 0 && e.([]float64)[0] < 0 {
				t.Error("Expected non negative elements")
			}
			if i == 1 && e.([]float64)[0] >= 0 {
				t.Error("Expected negative elements")
			}
		}
	}
}

func TestClust_Loss(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = real.RealSpace{}

	var s = 0.
	for _, e := range testPoints {
		f := e.([]float64)[0]
		if f < 0 {
			f += 1
		}
		s += f * f
	}

	var l = clust.Loss(testPoints, sp, 2.)

	if s != l {
		t.Error("Expected", s, "got", l)
	}
}

func TestDBA(t *testing.T) {
	var sp = real.RealSpace{}

	var _, err = core.DBA([]core.Elemt{}, sp)

	if err == nil {
		t.Error("Expected empty error")
	}

	var elemts = []core.Elemt{[]float64{2.}, []float64{4.}}

	var dba, _ = core.DBA(elemts, sp)
	if e := dba.([]float64)[0]; e != 3. {
		t.Error("Expected 3 got", e)
	}

	elemts = testPoints
	dba, _ = core.DBA(elemts, sp)
	var s = 0.
	for i := 0; i < len(elemts); i++ {
		s += (elemts[i]).([]float64)[0]
	}

	var m = s / float64(len(elemts))
	var e = dba.([]float64)[0]
	if m != e {
		t.Error("Expected", m, "got", e)
	}

	var elemts2 = make([]core.Elemt, len(elemts))
	for i := range elemts {
		e := (elemts[i]).([]float64)[0]
		elemts2[i] = []float64{e, 2 * e}
	}

	var dba2, _ = core.DBA(elemts2, sp)
	var ee = dba2.([]float64)
	if ee[0] != m || ee[1] != 2*m {
		t.Error("Expected [m m] got", ee)
	}
}

func TestClust_Initializer(t *testing.T) {
	var clust = core.Clust{
		[]float64{0.},
		[]float64{-1.},
	}
	var sp = real.RealSpace{}
	var c, _ = clust.Initializer(2, testPoints, sp, nil)

	if !reflect.DeepEqual(clust, c) {
		t.Error("Expected identity")
	}
}

func TestClust_Empty(t *testing.T) {
	var testPanic = func() {
		if x := recover(); x == nil {
			t.Error("Expected error")
		}
	}

	func() {
		defer testPanic()

		core.Clust{}.AssignAll(testVectors, real.RealSpace{})
	}()
}
