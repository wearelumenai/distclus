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
	Value ClustStatus
	Error error
}

// Alive check if status is running without error
func (status OCStatus) Alive() bool {
	return status.Value == Ready || status.Playing()
}

// Playing check if status is running or idle
func (status OCStatus) Playing() bool {
	return status.Value == Running || status.Value == Idle
}

// NewOCStatus returns ocstatus with specific cluststatus
func NewOCStatus(status ClustStatus) OCStatus {
	return OCStatus{Value: status}
}

// NewOCStatusError returns new errored ocstatus
func NewOCStatusError(err error) OCStatus {
	return OCStatus{
		Value: Finished,
		Error: err,
	}
}
