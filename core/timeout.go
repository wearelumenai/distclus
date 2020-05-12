package core

import (
	"sync"
	"time"
)

// Timeout interface for managing timeout algorithm
type Timeout interface {
	Disable()
	Enabled() bool
}

// timeout algorithm timeout structure
type timeout struct {
	duration     time.Duration
	enabled      bool
	ack          <-chan bool
	interruption func(OCStatus) error
	mutex        sync.RWMutex
	finishing    Finishing
	ocm          OCModel
}

func (timeout *timeout) Enabled() bool {
	timeout.mutex.RLock()
	defer timeout.mutex.RUnlock()
	return timeout.enabled
}

// InterruptionTimeout process
func InterruptionTimeout(duration time.Duration, interruption func(OCStatus) error) (result Timeout) {
	result = &timeout{
		duration:     duration,
		enabled:      true,
		interruption: interruption,
	}
	go result.(*timeout).interrupt()
	return
}

// WaitTimeout process. Return true if timedout
func WaitTimeout(finishing Finishing, duration time.Duration, ocm OCModel, ack <-chan bool) error {
	var t = timeout{
		duration:  duration,
		ack:       ack,
		ocm:       ocm,
		finishing: finishing,
	}
	return t.wait()
}

func (timeout *timeout) interrupt() {
	time.Sleep(timeout.duration)
	if timeout.Enabled() {
		timeout.interruption(OCStatus{Status: Finished, Error: ErrTimeout})
	}
}

func (timeout *timeout) isElapsedIter() bool {
	return IsFinished(timeout.finishing, timeout.ocm)
}

func (timeout *timeout) isAlive(lastTime time.Time) bool {
	return timeout.duration == 0 || time.Now().Before(lastTime)
}

func (timeout *timeout) wait() (err error) {
	err = ErrTimeout
	var step = 1 * time.Millisecond
	var elapsedTime = time.Now().Add(timeout.duration)
	for err == ErrTimeout && timeout.isAlive(elapsedTime) {
		select {
		case <-timeout.ack:
			err = nil
			break
		default:
			if timeout.isElapsedIter() {
				err = ErrElapsedIter
			}
		}
		time.Sleep(step)
	}
	return
}

func (timeout *timeout) Disable() {
	timeout.mutex.Lock()
	timeout.enabled = false
	timeout.mutex.Unlock()
}
