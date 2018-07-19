package kalman

import (
	"errors"
	"fmt"

	"github.com/gonum/matrix/mat64"
)

type KalmanFilter struct {
	p    *mat64.Dense //
	r    *mat64.Dense // Covariance of the Observation Noise
	i    *mat64.Dense
	xHat *mat64.Dense // System State Matrix
}

func New(covError, obsNoise *mat64.Dense) (*KalmanFilter, error) {

	rP, cP := covError.Dims()
	if rP != cP {
		return nil, errors.New("P dimenssion mismatch")
	}

	identity := mat64.NewDense(rP, cP, nil)
	for i := 0; i < rP; i++ {
		identity.Set(i, i, 1.0)
	}

	xHat := mat64.NewDense(rP, 1, nil)

	return &KalmanFilter{
		p:    covError,
		r:    obsNoise,
		i:    identity,
		xHat: xHat,
	}, nil
}

func (f *KalmanFilter) Init(x *mat64.Vector, v mat64.Matrix) error {

	return nil
}

func (f *KalmanFilter) Update(obs, modelInput *mat64.Dense, sigma float64) error {
	var y, s, k, xHat, p mat64.Dense
	var t0, t1, t2, t3, t4, t5, t6, t7, t8 mat64.Dense

	rP, cP := f.p.Dims()
	si := mat64.NewDense(rP, cP, nil)
	for r := 0; r < rP; r++ {
		for c := 0; c < cP; c++ {
			si.Set(r, c, sigma)
		}
	}

	t0.Add(f.p, si)
	f.p = &t0

	t1.Mul(modelInput, f.xHat)
	y.Sub(obs, &t1)

	t2.Mul(modelInput, f.p)
	t3.Mul(&t2, modelInput.T())
	s.Add(f.r, &t3)

	t4.Mul(f.p, modelInput.T())
	if err := t5.Inverse(&s); err != nil {
		fmt.Println(err)
	}
	k.Mul(&t4, &t5)

	t6.Mul(&k, &y)
	xHat.Add(f.xHat, &t6)
	f.xHat = &xHat

	t7.Mul(&k, modelInput)
	t8.Sub(f.i, &t7)
	p.Mul(&t8, f.p)
	f.p = &p

	return nil
}

func (f *KalmanFilter) Predict(modelInput *mat64.Dense) (mat64.Dense, error) {
	var t1 mat64.Dense

	t1.Mul(modelInput, f.xHat)
	return t1, nil
}

func (f *KalmanFilter) GetStat() mat64.Dense {
	return *f.xHat
}

func (f *KalmanFilter) GetError() mat64.Dense {
	return *f.p
}
