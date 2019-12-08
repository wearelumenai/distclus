package core

// ClustStatus integer type
type ClustStatus int64

// ClustStatus const values
const (
	Created       ClustStatus = iota
	Initializing              // initializing
	Ready                     // ready to run
	Running                   // used when algorithm run
	Idle                      // paused by user
	Sleeping                  // iteration frequency brake
	Stopping                  // stopped by user
	Failed                    // if an error occured during execution
	Succeed                   // if clustering succeed
	Reconfiguring             // reconfiguration in progress
	Interrupted               // after stopping the algorithm
)

var names = []string{
	"Created", "Initializing", "Ready",
	"Running", "Idle", "Sleeping", "Stopping",
	"Failed", "Succeed", "Reconfiguring", "Interrupted",
}

// String display value message
func (clustStatus ClustStatus) String() string {
	return names[int(clustStatus)]
}
