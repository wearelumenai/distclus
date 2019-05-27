package streaming

import (
	"golang.org/x/exp/rand"
	"time"
)

// Conf represents the cofiguration of a streaming algorithm.
type Conf struct {
	BufferSize int
	B          float64
	Lambda     float64
	RGen       *rand.Rand
}

// SetConfigDefaults applies default values to the given configuration.
func SetConfigDefaults(conf *Conf) {
	if conf.BufferSize == 0 {
		conf.BufferSize = 100
	}
	if conf.B == 0. {
		conf.B = .95
	}
	if conf.Lambda == 0. {
		conf.Lambda = 3.
	}
	if conf.RGen == nil {
		conf.RGen = rand.New(rand.NewSource(uint64(time.Now().Nanosecond())))
	}
}

// Verify checks if the given configuration is valid.
func Verify(conf Conf) {

}
