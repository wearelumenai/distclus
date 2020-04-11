package core_test

import (
	"github.com/wearelumenai/distclus/v0/internal/test"
	"github.com/wearelumenai/distclus/v0/pkg/core"
	"github.com/wearelumenai/distclus/v0/pkg/euclid"
	"testing"
)

func TestClust_ParMapLabel(t *testing.T) {
	var data = make([]core.Elemt, 0, len(test.Vectors)*20)
	for i := 0; i < 20; i++ {
		data = append(data, test.Vectors...)
	}
	var centroids = core.Clust(test.Vectors[0:3])

	for i := 1; i < 100; i++ {
		var seqLabels, _ = centroids.MapLabel(data, euclid.Space{})
		var parLabels, _ = centroids.ParMapLabel(data, euclid.Space{}, i)

		test.AssertArrayEqual(t, seqLabels, parLabels)
	}
}
