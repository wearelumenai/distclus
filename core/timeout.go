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
	interruption func(ClustStatus, error) error
	ack          <-chan bool
	step         time.Duration
	mutex        sync.RWMutex
}

func (timeout *timeout) Enabled() bool {
	timeout.mutex.RLock()
	defer timeout.mutex.RUnlock()
	return timeout.enabled
}

// InterruptionTimeout process
func InterruptionTimeout(duration time.Duration, interruption func(ClustStatus, error) error) (result Timeout) {
	result = &timeout{
		duration:     duration,
		enabled:      true,
		interruption: interruption,
	}
	go result.(*timeout).interrupt()
	return
}

// WaitTimeout process. Return true if timedout
func WaitTimeout(duration time.Duration, step time.Duration, ack <-chan bool) bool {
	var t = timeout{
		duration: duration,
		ack:      ack,
		step:     step,
	}
	return t.wait()
}

func (timeout *timeout) interrupt() {
	time.Sleep(timeout.duration)
	if timeout.Enabled() {
		timeout.interruption(Failed, ErrTimeout)
	}
}

func (timeout *timeout) wait() (timedout bool) {
	if timeout.duration == 0 {
		<-timeout.ack
	} else {
		timedout = true
		var elapsedTime = time.Now().Add(timeout.duration)
		for time.Now().Before(elapsedTime) {
			time.Sleep(timeout.step)
			select {
			case <-timeout.ack:
				timedout = false
				break
			default:
			}
		}
	}
	return timedout
}

func (timeout *timeout) Disable() {
	timeout.mutex.Lock()
	timeout.enabled = false
	timeout.mutex.Unlock()
}
