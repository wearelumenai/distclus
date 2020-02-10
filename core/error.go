package core

import "errors"

// ErrNotStarted fired if centroids or runtimefigures asked without running the algorithm
var ErrNotStarted = errors.New("clustering not started")

// ErrNotRunning raised while algorithm status equals Created, Ready or Failed
var ErrNotRunning = errors.New("Algorithm is not running")

// ErrRunning raised while algorithm status equals Running, Idle or Sleeping
var ErrRunning = errors.New("Algorithm is running")

// ErrStopping raised while algorithm status equals stopping
var ErrStopping = errors.New("Algorithm is stopping")

// ErrAlreadyCreated raised if algorithm is already created
var ErrAlreadyCreated = errors.New("Algorithm is already created")

// ErrNotIdle idle status is asked and not setted
var ErrNotIdle = errors.New("Algorithm is not idle")

// ErrIdle raised if algo is idle
var ErrIdle = errors.New("Algorithm is idle")

// ErrWaiting raised if algo is waiting
var ErrWaiting = errors.New("Algorithm is waiting")

// ErrReconfiguring raised if algo is reconfiguring
var ErrReconfiguring = errors.New("Algorithm is reconfiguring")

// ErrInfiniteIterations occure when static execution is asked with infinite iterations
var ErrInfiniteIterations = errors.New("Infinite iterations in static mode")

// ErrTimeout is returned when an error occurs
var ErrTimeout = errors.New("algorithm timed out")

// ErrNeverEnd raised when wait method is called while the algorithm will never end
var ErrNeverEnd = errors.New("algorithm can not end. Specify core.Conf.Iter, core.Conf.IterPerData or core.Conf.DataPerIter for allowing your algorithm to sleep")

// ErrSleeping raised when sleeping methods is called while the algorithm will never sleep
var ErrSleeping = errors.New("algorithm is sleeping")

// ErrStopped raised when sleeping methods is called while the algorithm is sttopped
var ErrStopped = errors.New("algorithm is stopped")

// ErrClosed raised when algo is closed
var ErrClosed = errors.New("algorithm is closed")

// ErrNotIterate raised when play is called while algo can not iterate
var ErrNotIterate = errors.New("algorithm can not iterate. Check iterations and dataPerIter conditions")
