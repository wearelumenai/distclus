package core_test

import (
	"distclus/core"
	"testing"
)

func TestGetChunk(t *testing.T) {
	for offset := 1; offset < len(testPoints)*2; offset++ {
		for chunkNumber := 0; chunkNumber <= len(testPoints)/offset + 1; chunkNumber++ {
			var parts = core.GetChunk(chunkNumber, offset, testPoints)
			l := computeChunkSize(offset, chunkNumber)
			if len(parts) != l {
				t.Error("Expected", l, "elements got", len(parts))
			}
		}
	}
}

func computeChunkSize(offset int, chunkNumber int) int {
	var l = offset
	if chunkNumber*offset+offset >= len(testPoints) {
		l = len(testPoints) - chunkNumber*offset
	}
	if l < 0 {
		l = 0
	}
	return l
}
