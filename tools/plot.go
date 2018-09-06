package tools

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"image/color"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"distclus/core"
)

func PlotClust(clust core.Clust, batch []core.Elemt, space core.Space, title, xLab, yLab, png string) {
	var clusts = clust.AssignAll(batch, space)
	k := len(clust)
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = title
	p.X.Label.Text = xLab
	p.Y.Label.Text = yLab
	pts := make([]interface{}, k)
	for i := range pts {
		xys := make(plotter.XYs, len(clusts[i]))
		pts[i] = xys
		for j, e := range clusts[i] {
			ee := e.([]float64)
			xys[j].X = ee[0]
			xys[j].Y = ee[1]
		}
	}
	cpts := make(plotter.XYs, k)
	for i := range pts {
		c := clust[i].([]float64)
		cpts[i].X = c[0]
		cpts[i].Y = c[1]
	}
	scpts, err := plotter.NewScatter(cpts)
	if err != nil {
		panic(err)
	}
	scpts.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	scpts.Shape = draw.CrossGlyph{}
	plotutil.AddScatters(p, pts...)
	p.Add(scpts)
	if err := p.Save(10*vg.Centimeter, 10*vg.Centimeter, png + ".png"); err != nil {
		panic(err)
	}
}
