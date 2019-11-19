package core

// ClustStatus integer type
type ClustStatus = int64

// ClustStatus const values
const (
	Created  ClustStatus = iota
	Ready                // ready to run
	Running              // used when algorithm run
	Idle                 // paused by user
	Sleeping             // waiting for data to process
	Failed               // if an error occured during execution
	Stopping             // stopped by user
)
