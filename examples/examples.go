package examples

import (
	"os"
	"bufio"
	"strings"
	"strconv"
	"distclus/core"
	"time"
	"distclus/algo"
	"distclus/tools"
)

func s1Parser(path string) []core.Elemt {
	inFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer inFile.Close()
	scanner := bufio.NewScanner(inFile)
	scanner.Split(bufio.ScanLines)
	var res []core.Elemt
	for scanner.Scan() {
		var split = strings.Split(scanner.Text(), "    ")
		var x, _ = strconv.ParseFloat(split[1], 64)
		var y, _ = strconv.ParseFloat(split[2], 64)
		res = append(res, []float64{x, y})
	}
	return res
}

func mcmcClust(space core.RealSpace, batch []core.Elemt) {
	var mcmcConf = algo.MCMCConf{
		Dim:         2, FrameSize: 8, B: 100, Amp: 1,
		Norm:        2, Nu: 1, InitK: 3, McmcIter: 100,
		InitIter:    1, Space: space,
		Initializer: algo.KmeansPPInitializer,
		Seed:        uint64(time.Now().UTC().Unix()),
	}
	var distrib, _ = algo.NewMultivT(algo.MultivTConf{mcmcConf})
	var mcmc = algo.NewMCMC(mcmcConf, &distrib)
	for _, elt := range batch {
		mcmc.Push(elt)
	}
	mcmc.Run()
	mcmc.Close()
	var centers, _ = mcmc.Centroids()
	k := len(centers)
	println("centers: ", k)
	tools.PlotClust(centers, batch, space, "s1MCMC", "X", "Y", "s1MCMC")
}

func kmClust(space core.RealSpace, batch []core.Elemt) {
	var km = algo.NewKMeans(algo.KMeansConf{16, 100, space}, algo.KmeansPPInitializer)
	for _, e := range batch {
		km.Push(e)
	}
	km.Run()
	km.Close()
	var centers, _ = km.Centroids()
	tools.PlotClust(centers, batch, space, "s1MCMC", "X", "Y", "s1Kmeans")
}

func main() {
	var batch = s1Parser("examples/data/s1.txt")
	var space = core.RealSpace{}
	kmClust(space, batch)
	mcmcClust(space, batch)
}
