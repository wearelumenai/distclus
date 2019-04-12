package streaming_test

import (
	"distclus/streaming"
	"testing"
)

func Test_SetDefaultConfig(t *testing.T) {
	var conf = streaming.Conf{}
	streaming.SetConfigDefaults(&conf)
	if conf.BufferSize != 100 {
		t.Error("expected 100")
	}
	if conf.B != .95 {
		t.Error("expected 1")
	}
	if conf.Lambda != 3. {
		t.Error("expected 1")
	}
	if conf.RGen == nil {
		t.Error("expected non nil")
	}
}
