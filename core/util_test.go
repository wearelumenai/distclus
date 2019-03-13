package core_test

import (
	"distclus/core"
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
		var process = func(part []core.Elemt, start int, end int, rank int) {
			copy(result[start:end], part)
		}
		core.Par(process, data, degree)
		if !reflect.DeepEqual(data, result) {
			t.Error("par error")
		}
	}
}
