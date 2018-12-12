package core

// Elemt interface that can be used in a clustering algorithm
type Elemt interface{}

// Space operations needed for clustering a set of elements
type Space interface {
	Dist(elemt1, elemt2 Elemt) float64
	Combine(elemt1 Elemt, weight1 int, elemt2 Elemt, weight2 int)
	Copy(elemt Elemt) Elemt
	Dim(data []Elemt) int
}

// SpaceConf is a space configuration interface
type SpaceConf interface{}
