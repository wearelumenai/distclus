package algo

import (
	"testing"
	"distclus/core"
)

func TestBuffer_Push(t *testing.T) {
	var buf = newBuffer(make([]core.Elemt, 0))

	for i:=0; i<128; i++ {
		buf.push([]float64{float64(i),1.,2.,4.})
	}

	if l:=len(buf.Data); l != 128 {
		t.Error("Expected 128 got", l)
	}

	for i:=0; i<128; i++ {
		if j:= (buf.Data)[i].([]float64)[0]; float64(i)!=j {
			t.Error("Expected", i, "got", j)
		}
	}
}

func TestBuffer_Apply(t *testing.T) {
	var buf = newBuffer(make([]core.Elemt, 0))

	for i:=0; i<128; i++ {
		buf.push([]float64{float64(i),1.,2.,4.})
	}

	if l:=len(buf.Data); l != 128 {
		t.Error("Expected 128 got", l)
	}

	buf.setAsync()

	for i:=0; i<128; i++ {
		buf.push([]float64{float64(i),1.,2.,4.})
	}

	if l:=len(buf.Data); l != 128 {
		t.Error("Expected 128 got", l)
	}

	buf.apply()

	if l:=len(buf.Data); l != 256 {
		t.Error("Expected 256 got", l)
	}
}
