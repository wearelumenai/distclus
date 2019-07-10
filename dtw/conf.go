// Package dtw allows to computes DTW distance based clusters.
package dtw

// Conf defines series configuration
type Conf struct {
	InnerSpace PointSpace
	Window     int
}
