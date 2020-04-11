package streaming

import (
	"github.com/wearelumenai/distclus/v0/pkg/core"
	"time"

	"golang.org/x/exp/rand"
)

// Conf represents the cofiguration of a streaming algorithm.
type Conf struct {
	core.Conf
	BufferSize int
	Mu         float64
	Sigma      float64
	OutRatio   float64
	OutAfter   int
	RGen       *rand.Rand
}

// SetConfigDefaults applies default values to the given configuration.
func (conf *Conf) SetConfigDefaults() {
	if conf.BufferSize == 0 {
		conf.BufferSize = 100
	}
	if conf.Mu == 0. {
		conf.Mu = .5
	}
	if conf.OutRatio == 0. {
		conf.OutRatio = 2.
	}
	if conf.OutAfter == 0 {
		conf.OutAfter = 5
	}
	if conf.RGen == nil {
		conf.RGen = rand.New(rand.NewSource(uint64(time.Now().Nanosecond())))
	}
	if conf.Sigma == 0 {
		conf.Sigma = 0.1
	}
}

// Verify checks if the given configuration is valid.
func (conf *Conf) Verify() {
	conf.SetConfigDefaults()
	conf.Conf.Verify()
	if conf.OutAfter < 2 {
		panic("OutAfter should be greater than 1")
	}
}
