package streaming_test

import (
	"distclus/internal/test"
	"testing"
)

func TestStreaming_ConfErrorIter(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = streamingConf
	conf.StreamingIter = -10
	conf.Verify()
}

func TestStreaming_ConfErrorMaxK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = streamingConf
	conf.InitK = 30
	conf.MaxK = 10
	conf.Verify()
}

func TestStreaming_ConfErrorK(t *testing.T) {
	defer test.AssertPanic(t)
	var conf = streamingConf
	conf.InitK = 0
	conf.Verify()
}
