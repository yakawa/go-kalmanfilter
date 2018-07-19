package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/yakawa/go-kalmanfilter"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

const (
	a = 5
	b = 3
	c = 2
)

func randUniform(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}

func main() {
	rand.Seed(time.Now().UnixNano())

	P := mat64.NewDense(3, 3, []float64{
		1000, 0, 0,
		0, 1000, 0,
		0, 0, 1000,
	})
	R := mat64.NewDense(1, 1, []float64{0.1})

	kf, _ := kalman.New(P, R)

	obs := make([]float64, 365)
	x1 := make([]float64, 365)
	x2 := make([]float64, 365)
	a_x1 := make([]float64, 365)
	b_x2 := make([]float64, 365)
	c_ := make([]float64, 365)
	days := make([]float64, 365)

	for i := 0; i < 365; i++ {
		x1[i] = randUniform(-5.0, 5.0)
		x2[i] = randUniform(0.0, 3.0)
		if i <= 185 {
			obs[i] = a*x1[i] + b*x2[i] + c + randUniform(-1.0, 1.0)
		} else {
			obs[i] = (a+2)*x1[i] + b*x2[i] + c + randUniform(-1.0, 1.0)
		}
		obsM := mat64.NewDense(1, 1, []float64{obs[i]})
		system := mat64.NewDense(1, 3, []float64{x1[i], x2[i], 1.0})
		kf.Update(obsM, system, 0.0001)
		t := kf.GetStat()
		a_x1[i] = t.At(0, 0)
		b_x2[i] = t.At(1, 0)
		c_[i] = t.At(2, 0)
		days[i] = float64(i + 1)
	}

	p, err := plot.New()
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := plotutil.AddLinePoints(p,
		"X1", generatePoints(days, a_x1),
		"X2", generatePoints(days, b_x2),
		"C", generatePoints(days, c_),
	); err != nil {
		fmt.Println(err)
		return
	}

	if err := p.Save(15*vg.Inch, 4*vg.Inch, "sample.png"); err != nil {
		fmt.Println(err)
	}

}

func generatePoints(x, y []float64) plotter.XYs {
	pts := make(plotter.XYs, len(x))
	for i := range pts {
		pts[i].X = x[i]
		pts[i].Y = y[i]
	}

	return pts
}
