package main

import (
	"distclus/kmeans"
	"gopkg.in/alecthomas/kingpin.v2"
	"os"
	"encoding/csv"
	"strconv"
	"fmt"
	"io"
	"distclus/core"
	"distclus/mcmc"
	"distclus/real"
	"golang.org/x/exp/rand"
	"log"
)

var (
	app = kingpin.New("core", "Go clustering")

	dtype = app.Flag("type", "Data type(real).").
		Short('t').Default("real").Enum("real")
	norm = app.Flag("norm", "Distance normalisation coefficient.").
		Short('n').Default("2").Float()
	seed = app.Flag("seed", "Seed for random initializer(time by default ~ -1).").
		Short('s').Default("-1").Int()
	fdata = app.Flag("data", "Data file path(supported format: CSV), if not set read from stdin.").
		Short('f').String()
	olabels = app.Flag("out_labels", "Filename where to print labels in csv, if not set printed in stdout.").
		Short('l').String()
	ocenters = app.Flag("out_centers", "Filename where to print centers in csv, if not set printed in stdout.").
		Short('c').String()

	mcmcCmd  = app.Command("mcmc", "Compute an MCMC clustering.")
	mcmcB = mcmcCmd.Flag("mcmc_b", "b parameter").
		Short('B').Default("100").Float()
	mcmcAmp = mcmcCmd.Flag("mcmc_amp", "amp MCMC parameter.").
		Short('A').Default("10").Float()
	mcmcR = mcmcCmd.Flag("mcmc_r", "Radius of the circumscribed ball.").
		Short('R').Default("1").Float()
	mcmcNu = mcmcCmd.Flag("mcmc_nu", "Number of degrees of freedom.").
		Short('D').Default("3").Float()
	mcmcInitK = mcmcCmd.Flag("mcmc_initk", "k initialisation value.").
		Short('K').Default("8").Int()
	mcmcFrameSize = mcmcCmd.Flag("mcmc_framesize", "Frame size to consider in data history(default -1 ~ data set len).").
		Short('F').Default("-1").Int()
	mcmcInitializer = mcmcCmd.Flag("mcmc_initializer", "Algorithm initializer(random, kmeans++).").
		Default("random").Short('i').Enum("random", "kmeans++")
	mcmcIter = mcmcCmd.Flag("mcmc_iter", "Max iteration of mcmc clustering.").
		Short('I').Default("200").Int()
	mcmcInitIter = mcmcCmd.Flag("mcmc_init_iter", "Number of initialisation iteration.").
		Default("0").Int()
)

func main() {
	kingpin.CommandLine.HelpFlag.Short('h')
	parse := kingpin.MustParse(app.Parse(os.Args[1:]))
	switch parse {
	case mcmcCmd.FullCommand():
		runMcmc()
	}
}

func runMcmc() {
	var data []core.Elemt
	var distrib mcmc.MCMCDistrib
	var initializer = parseInitializer(*mcmcInitializer)

	var mcmcConf = mcmc.MCMCConf{
		AlgoConf: core.AlgoConf {
			InitK:     *mcmcInitK,
			FrameSize: *mcmcFrameSize,
		},
		B:         *mcmcB,
		Amp:       *mcmcAmp,
		R:         *mcmcR,
		Norm:      *norm,
		Nu:        *mcmcNu,
		McmcIter:  *mcmcIter,
		InitIter:  *mcmcInitIter,
	}

	if *seed > -1 {
		mcmcConf.RGen = rand.New(rand.NewSource(uint64(*seed)))
	}

	switch *dtype {
	case "real":
		mcmcConf.Space = real.RealSpace{}
		data, mcmcConf.Dim = parseFloatCsv(fdata)
		// because the configuration is copied it must not be modified after object initialization
		if mcmcConf.FrameSize < 1 {
			mcmcConf.FrameSize = len(data)
		}
		distrib = mcmc.NewMultivT(mcmc.MultivTConf{mcmcConf})
	}


	var algo = mcmc.NewParMCMC(mcmcConf, distrib, initializer, nil)

	log.Println(fmt.Sprintf("Add data to algo model : %v obs.", len(data)))
	for _, elt := range data {
		algo.Push(elt)
	}

	algo.Run(false)
	algo.Close()

	var centers, _ = algo.Centroids()
	var labels = make([]int, len(data))
	var abstract = make([]int, len(centers))
	for i := range labels {
		_, labels[i], _ = centers.Assign(data[i], mcmcConf.Space)
		abstract[labels[i]] += 1
	}

	log.Println(fmt.Sprintf("Cluster cards : %v", abstract))
	log.Println(fmt.Sprintf("Loss : %v", centers.Loss(data, mcmcConf.Space, mcmcConf.Norm)))
	log.Println(fmt.Sprintf("Acceptation ratio : %v", algo.AcceptRatio()))

	printLabels(labels, olabels)
	printCenters(centers, ocenters)
}

func printLabels(res []int, out *string) {
	var o io.Writer
	if len(*out) != 0 {
		var f, err = os.Create(*out)
		if err != nil {
			panic(err)
		}
		o = f
		defer f.Close()
	} else {
		o = os.Stdout
	}
	var writer = csv.NewWriter(o)
	defer writer.Flush()
	for _, label := range res {
		err := writer.Write([]string{strconv.Itoa(label)})
		if err != nil {
			panic(fmt.Sprintf("Cannot write to file %s", err))
		}
	}
}

func printCenters(res core.Clust, out *string) {
	var o io.Writer
	if len(*out) != 0 {
		var f, err = os.Create(*out)
		if err != nil {
			panic(err)
		}
		o = f
		defer f.Close()
	} else {
		o = os.Stdout
	}
	var writer = csv.NewWriter(o)
	defer writer.Flush()
	for _, label := range res {
		err := writer.Write([]string{fmt.Sprint(label)})
		if err != nil {
			panic(fmt.Sprintf("Cannot write to file %s", err))
		}
	}
}

func parseInitializer(init string) core.Initializer {
	var initializer core.Initializer

	switch init {
	case "random":
		initializer = kmeans.KMeansPPInitializer
	case "kmeans++":
		initializer = kmeans.RandInitializer
	}

	return initializer
}

func parseFloatCsv(in *string) ([]core.Elemt, int) {
	var i *os.File
	if len(*in) != 0 {
		var f, err = os.Open(*in)
		if err != nil {
			panic(err)
		}
		i = f
	} else {
		i = os.Stdin
	}
	defer i.Close()

	var reader = csv.NewReader(i)
	reader.Comma = ','
	rows, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}
	if len(rows) == 0 {
		panic("file empty")
	}
	var parsed []core.Elemt
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
