package streaming_test

import (
	"testing"

	"go.lumenai.fr/distclus/v0/internal/test"
	"go.lumenai.fr/distclus/v0/streaming"
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
	if conf.Sigma != 0.1 {
		t.Error("expected 0.1")
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
	conf.Verify()
}
