package core

// Node interface representing a clusterisable entity.
type Elemt interface{}

// Space in which are defined nodes.
type Space interface {
	Dist(elemt1, elemt2 Elemt) float64
	Combine(elemt1 Elemt, weight1 int, elemt2 Elemt, weight2 int)
	Copy(elemt Elemt) Elemt
}
