package main

import (
	"os"
	"bufio"
	"strings"
	"strconv"
	"github.com/blqn/clustering-go"
	"time"
)

func s1Parser(path string) []clustering_go.Elemt {
	inFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer inFile.Close()
	scanner := bufio.NewScanner(inFile)
	scanner.Split(bufio.ScanLines)
	var res []clustering_go.Elemt
	for scanner.Scan() {
		var split = strings.Split(scanner.Text(), "    ")
		var x, _ = strconv.ParseFloat(split[1], 64)
		var y, _ = strconv.ParseFloat(split[2], 64)
		res = append(res, []float64{x, y})
	}
	return res
}

func mcmcClust(space clustering_go.RealSpace, batch []clustering_go.Elemt) {
	var mcmcConf = clustering_go.MCMCConf{
		Dim:         2, FrameSize: 8, B: 100, Amp: 1,
		Norm:        2, Nu: 1, InitK: 3, McmcIter: 100,
		InitIter:    1, Space: space,
		Initializer: clustering_go.KmeansPPInitializer,
		Seed:        uint64(time.Now().UTC().Unix()),
	}
	var distrib, _ = clustering_go.NewMultivT(clustering_go.MultivTConf{mcmcConf})
	var mcmc = clustering_go.NewMCMC(mcmcConf, &distrib)
	for _, elt := range batch {
		mcmc.Push(elt)
	}
	mcmc.Run()
	mcmc.Close()
	var centers, _ = mcmc.Centroids()
	k := len(*centers.Centers())
	println("centers: ", k)
	clustering_go.PlotClust(centers, &batch, space, "s1MCMC", "X", "Y", "s1MCMC")
}

func kmClust(space clustering_go.RealSpace, batch []clustering_go.Elemt) {
	k := 16
	var km = clustering_go.NewKMeans(k, 100, space, clustering_go.KmeansPPInitializer)
	for _, e := range batch {
		km.Push(e)
	}
	km.Run()
	km.Close()
	var centers, _ = km.Centroids()
	clustering_go.PlotClust(centers, &batch, space, "s1MCMC", "X", "Y", "s1Kmeans")
}

func main() {
	var batch = s1Parser("examples/data/s1.txt")
	var space = clustering_go.RealSpace{}
	kmClust(space, batch)
	mcmcClust(space, batch)
}
