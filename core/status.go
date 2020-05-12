package core

// ClustStatus integer type
type ClustStatus int64

// ClustStatus const values
const (
	Created      ClustStatus = iota
	Initializing             // initializing
	Ready                    // ready to run
	Running                  // used when algorithm is playing
	Idle                     // paused by user
	Finished                 // when the algorithm has finished
)

var names = []string{
	"Created", "Initializing", "Ready", // starting
	"Running", "Idle", // clustering
	"Finished", // converged or failed
}

// String display value message
func (clustStatus ClustStatus) String() string {
	return names[int(clustStatus)]
}

// StatusNotifier for being notified by Online clustering change status
type StatusNotifier = func(OnlineClust, OCStatus)

// OCStatus describes Online Clustering status with ClustStatus and error
type OCStatus struct {
	Status ClustStatus
	Error  error
}

// Alive check if status is running without error
func (status OCStatus) Alive() bool {
	return status.Status >= Running && status.Error == nil
}
