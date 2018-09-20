package kmeans

import "distclus/core"

type SeqKMeansSupport struct {
	config KMeansConf
	buffer *core.Buffer
}

func (support SeqKMeansSupport) Iterate(clust core.Clust) core.Clust {

	var result, _ = clust.AssignDBA(support.buffer.Data, support.config.Space)

	return support.buildResult(clust, result)
}

func (support SeqKMeansSupport) buildResult(clust core.Clust, result core.Clust) core.Clust {
	for i := 0; i < len(result); i++ {
		if result[i] == nil {
			result[i] = clust[i]
		}
	}

	return result
}
