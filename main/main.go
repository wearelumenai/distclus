package main

import (
	"gopkg.in/alecthomas/kingpin.v2"
	"os"
	"github.com/blqn/clustering-go"
	"time"
	"encoding/csv"
	"strconv"
	"fmt"
)

var (
	app = kingpin.New("distclus", "Go clustering")

	dtype = app.Flag("type", "Data type").
		Short('t').Default("real").Enum("real")
	norm = app.Flag("norm", "distance normalisation coefficient").
		Short('n').Default("2").Float()
	seed = app.Flag("seed", "seed for random initializer(time by default ~ -1)").
		Short('s').Default("-1").Int()
	fdata = app.Flag("data", "data file path(supported format: CSV)").
		Short('f').Required().String()

	mcmc  = app.Command("mcmc", "Compute an MCMC")
	mcmcB = mcmc.Flag("mcmc_b", "b parameter").
		Short('B').Default("100").Float()
	mcmcAmp = mcmc.Flag("mcmc_amp", "amp parameter").
		Short('A').Default("1").Float()
	mcmcNu = mcmc.Flag("mcmc_nu", "degrees of freedom number").
		Short('D').Default("2").Float()
	mcmcInitK = mcmc.Flag("mcmc_initk", "k initialisation value").
		Short('K').Default("8").Int()
	mcmcFrameSize = mcmc.Flag("mcmc_framesize", "framesize to consider in data history").
		Short('F').Default("-1").Int()
	mcmcInitializer = mcmc.Flag("mcmc_initializer", "Algorithm initializer").
		Default("random").Short('i').Enum("random", "kmeans++")
	mcmcIter = mcmc.Flag("mcmc_iter", "max iteration of mcmc clustering").
		Short('I').Default("-1").Int()
	mcmcInitIter = mcmc.Flag("mcmc_init_iter", "number of initialisation iteration").
		Default("1").Int()
)

func main() {
	kingpin.CommandLine.HelpFlag.Short('h')
	parse := kingpin.MustParse(app.Parse(os.Args[1:]))
	println(parse)
	switch parse {
	case mcmc.FullCommand():
		runMcmc()
	}
}
func runMcmc() {
	var space distclus.Space
	var data []distclus.Elemt
	var dim int
	var initializer = parseInitializer(*mcmcInitializer)
	switch *dtype {
	case "real":
		space = distclus.RealSpace{}
		data, dim = parseFloatCsv()
	}
	if *seed == -1 {
		*seed = int(time.Now().UTC().Unix())
	}
	if *mcmcFrameSize < 1 {
		*mcmcFrameSize = len(data)
	}
	var mcmcConf = distclus.MCMCConf{
		Dim:         dim,
		FrameSize:   *mcmcFrameSize,
		B:           *mcmcB,
		Amp:         *mcmcAmp,
		Norm:        *norm,
		Nu:          *mcmcNu,
		InitK:       *mcmcInitK,
		McmcIter:    *mcmcIter,
		InitIter:    *mcmcInitIter,
		Space:       space,
		Initializer: initializer,
		Seed:        uint64(*seed),
	}
	var distrib, ok = distclus.NewMultivT(distclus.MultivTConf{mcmcConf})
	if !ok{
		panic("can't initialize mcmc")
	}
	var mcmc = distclus.NewMCMC(mcmcConf, &distrib)
	for _, elt := range data {
		mcmc.Push(elt)
	}
	mcmc.Run()
	mcmc.Close()
	var centers, _ = mcmc.Centroids()
	k := len(*centers.Centers())
	println("centers: ", k)
	distclus.PlotClust(centers, &data, space, "s1MCMC", "X", "Y", "s1MCMC")
}

func parseInitializer(init string) (initializer func(k int, elemts []distclus.Elemt, space distclus.Space) (c distclus.Clust, err error)) {
	switch init {
	case "random":
		initializer = distclus.KmeansPPInitializer
	case "kmeans++":
		initializer = distclus.RandInitializer
	}
	return initializer
}

func parseFloatCsv() ([]distclus.Elemt, int) {
	file, err := os.Open(*fdata)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	var reader = csv.NewReader(file)
	reader.Comma = ','
	rows, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}
	if len(rows) == 0 {
		panic("file empty")
	}
	var parsed []distclus.Elemt
	var rlen = len(rows[0])
	if rlen == 0 {
		panic("first row is empty")
	}
	for i, row := range rows {
		if len(row) != rlen {
			panic(fmt.Sprintf("missing value at line %v", i))
		}
		var parsedRow = make([]float64, rlen)
		for i, val := range row {
			parsedRow[i], err = strconv.ParseFloat(val, 64)
			if err != nil {
				panic(err)
			}
		}
		parsed = append(parsed, parsedRow)
	}
	return parsed, rlen
}
