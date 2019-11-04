package streaming_test

import (
	"distclus/internal/test"
	"distclus/streaming"
	"testing"
)

func Test_SetDefaultConfig(t *testing.T) {
	var conf = streaming.Conf{}
	conf.SetConfigDefaults()
	if conf.BufferSize != 100 {
		t.Error("expected 100")
	}
	if conf.Mu != .5 {
		t.Error("expected .5")
	}
	if conf.Sigma != 0. {
		t.Error("expected 0.")
	}
	if conf.OutRatio != 2. {
		t.Error("expected 2.")
	}
	if conf.OutAfter != 5 {
		t.Error("expected 5")
	}
	if conf.RGen == nil {
		t.Error("expected non nil")
	}
}

func Test_VerifyConfig(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = streaming.Conf{OutAfter: 1}
	streaming.Verify(conf)
}
