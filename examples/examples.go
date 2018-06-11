package examples

import (
	"os"
	"bufio"
	"strings"
	"strconv"
	"github.com/blqn/clustering-go"
	"time"
)

func s1Parser(path string) []distclus.Elemt {
	inFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer inFile.Close()
	scanner := bufio.NewScanner(inFile)
	scanner.Split(bufio.ScanLines)
	var res []distclus.Elemt
	for scanner.Scan() {
		var split = strings.Split(scanner.Text(), "    ")
		var x, _ = strconv.ParseFloat(split[1], 64)
		var y, _ = strconv.ParseFloat(split[2], 64)
		res = append(res, []float64{x, y})
	}
	return res
}

func mcmcClust(space distclus.RealSpace, batch []distclus.Elemt) {
	var mcmcConf = distclus.MCMCConf{
		Dim:         2, FrameSize: 8, B: 100, Amp: 1,
		Norm:        2, Nu: 1, InitK: 3, McmcIter: 100,
		InitIter:    1, Space: space,
		Initializer: distclus.KmeansPPInitializer,
		Seed:        uint64(time.Now().UTC().Unix()),
	}
	var distrib, _ = distclus.NewMultivT(distclus.MultivTConf{mcmcConf})
	var mcmc = distclus.NewMCMC(mcmcConf, &distrib)
	for _, elt := range batch {
		mcmc.Push(elt)
	}
	mcmc.Run()
	mcmc.Close()
	var centers, _ = mcmc.Centroids()
	k := len(*centers.Centers())
	println("centers: ", k)
	distclus.PlotClust(centers, &batch, space, "s1MCMC", "X", "Y", "s1MCMC")
}

func kmClust(space distclus.RealSpace, batch []distclus.Elemt) {
	k := 16
	var km = distclus.NewKMeans(k, 100, space, distclus.KmeansPPInitializer)
	for _, e := range batch {
		km.Push(e)
	}
	km.Run()
	km.Close()
	var centers, _ = km.Centroids()
	distclus.PlotClust(centers, &batch, space, "s1MCMC", "X", "Y", "s1Kmeans")
}

func main() {
	var batch = s1Parser("examples/data/s1.txt")
	var space = distclus.RealSpace{}
	kmClust(space, batch)
	mcmcClust(space, batch)
}
