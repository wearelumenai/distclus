package streaming

import (
	"golang.org/x/exp/rand"
	"time"
)

type Conf struct {
	BufferSize int
	B          float64
	Lambda     float64
	RGen       *rand.Rand
}

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

func Verify(conf Conf) {

}
