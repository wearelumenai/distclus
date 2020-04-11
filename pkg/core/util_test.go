package core_test

import (
	"github.com/wearelumenai/distclus/v0/pkg/core"
	"reflect"
	"testing"
)

func Test_Par(t *testing.T) {
	for degree := 1; degree < 100; degree++ {
		var data = make([]core.Elemt, 2*3*5*7)
		var result = make([]core.Elemt, len(data))
		for i := range data {
			data[i] = i
		}
		var process = func(start int, end int, rank int) {
			copy(result[start:end], data[start:end])
		}
		core.Par(process, len(data), degree)
		if !reflect.DeepEqual(data, result) {
			t.Error("par error")
		}
	}
}
