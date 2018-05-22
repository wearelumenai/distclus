package clustering_go

// Node interface representing a clusterisable entity.
type node interface{}

// Space in which are defined nodes.
type space interface {
	dist(node1, node2 node) float64
	combine(node1 node, weight1 int, node2 node, weight2 int) node
}
