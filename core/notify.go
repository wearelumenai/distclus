package core

// Notifier structure
type Notifier struct {
	TTC      int64
	Callback func(Clust)
}
