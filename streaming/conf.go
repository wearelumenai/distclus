package streaming

import (
	"golang.org/x/exp/rand"
	"time"
)

// Conf represents the cofiguration of a streaming algorithm.
type Conf struct {
	BufferSize int
	Mu         float64
	Sigma      float64
	OutRatio   float64
	OutAfter   int
	RGen       *rand.Rand
}

// SetConfigDefaults applies default values to the given configuration.
func SetConfigDefaults(conf *Conf) {
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
}

// Verify checks if the given configuration is valid.
func Verify(conf Conf) {
	if conf.OutAfter < 2 {
		panic("OutAfter should be greater than 1")
	}
}
