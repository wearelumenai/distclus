package core_test

import (
	"distclus/core"
	"reflect"
	"testing"
)

func TestBuffer_Push(t *testing.T) {
	elemts := []core.Elemt{[]float64{1.2, 3.2}, []float64{1.2, 3.2}}
	var buf = core.NewDataBuffer(elemts, -1)

	if l := len(buf.Data()); l != 2 {
		t.Error("Expected 2 got", l)
	}

	if !reflect.DeepEqual(buf.Data(), elemts) {
		t.Error("Expected", elemts, "got", buf.Data())
	}

	for i := 2; i < 130; i++ {
		buf.Push([]float64{float64(i), 1., 2., 4.})
	}

	if l := len(buf.Data()); l != 130 {
		t.Error("Expected 130 got", l)
	}

	for i := 2; i < 130; i++ {
		if j := (buf.Data())[i].([]float64)[0]; float64(i) != j {
			t.Error("Expected", i, "got", j)
		}
	}
}

func TestBuffer_FrameMore(t *testing.T) {
	elemts := []core.Elemt{[]float64{1.2, 3.2}, []float64{1.2, 3.2}}
	var buf = core.NewDataBuffer(elemts, 50)

	if l := len(buf.Data()); l != 2 {
		t.Error("Expected 2 got", l)
	}

	if !reflect.DeepEqual(buf.Data(), elemts) {
		t.Error("Expected", elemts, "got", buf.Data())
	}

	for i := 0; i < 128; i++ {
		buf.Push([]float64{float64(i), 1., 2., 4.})
	}

	if l := len(buf.Data()); l != 50 {
		t.Error("Expected 50 got", l)
	}

	for i := 0; i < 50; i++ {
		if j := (buf.Data())[i].([]float64)[0]; j < 78 || j > 127 {
			t.Error("Expected 78<=j<128 got", j)
		}
	}
}

func TestBuffer_FrameLess(t *testing.T) {
	elemts := make([]core.Elemt, 120)
	for i := 0; i < 120; i++ {
		elemts[i] = [][]float64{{float64(i), 1.2, 3.2}, {1.2, 3.2}}
	}

	var buf = core.NewDataBuffer(elemts, 50)

	if l := len(buf.Data()); l != 50 {
		t.Error("Expected 50 got", l)
	}

	if !reflect.DeepEqual(buf.Data(), elemts[70:]) {
		t.Error("Expected", elemts, "got", buf.Data())
	}

	for i := 0; i < 128; i++ {
		buf.Push([]float64{float64(i), 1., 2., 4.})
	}

	if l := len(buf.Data()); l != 50 {
		t.Error("Expected 50 got", l)
	}

	for i := 0; i < 50; i++ {
		if j := (buf.Data())[i].([]float64)[0]; j < 78 || j > 127 {
			t.Error("Expected 78<=j<128 got", j)
		}
	}
}

func TestBuffer_Apply(t *testing.T) {
	var buf = core.NewDataBuffer(nil, -1)

	for i := 0; i < 128; i++ {
		buf.Push([]float64{float64(i), 1., 2., 4.})
	}

	if l := len(buf.Data()); l != 128 {
		t.Error("Expected 128 got", l)
	}

	buf.SetAsync()

	for i := 0; i < 128; i++ {
		buf.Push([]float64{float64(i), 1., 2., 4.})
	}

	if l := len(buf.Data()); l != 128 {
		t.Error("Expected 128 got", l)
	}

	buf.Apply()

	if l := len(buf.Data()); l != 256 {
		t.Error("Expected 256 got", l)
	}
}
