package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// or gate
var train_data = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

func sigmodf(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func cost_g(w1 float64, w2 float64, b float64) float64 {
	var result float64 = 0
	for i := 0; i < len(train_data); i++ {
		x1 := train_data[i][0]
		x2 := train_data[i][1]
		y := sigmodf(x1*w1 + x2*w2 + b)
		diff := y - train_data[i][2]

		result += diff * diff
		// fmt.Printf("actual: %f, expected: %f\n", y, train_data[i][1])
	}

	result /= float64(len(train_data))
	return result
}

func main() {
	// xor -> ~a&b + a&~b
	// for i := 0; i < 2; i++ {
	// 	for j := 0; j < 2; j++ {
	// 		notA := ^i
	// 		notB := ^j
	// 		fmt.Printf("%d ^ %d => %d\n", i, j, (notA&j + i&notB))
	// 	}
	// }
	// rand seed
	s1 := rand.NewSource(time.Now().UnixNano())
	// s1 := rand.NewSource(69)
	r := rand.New(s1)

	w1 := r.Float64()
	w2 := r.Float64()
	bias := r.Float64()
	// fmt.Printf("%f $ %f\n", w1, w2)
	rate := 1e-2
	eps := 1e-3

	for i := 0; i < 2000*1000; i++ {
		ccc := cost_g(w1, w2, bias)
		// fmt.Printf("%f\n", ccc)
		dw1 := (cost_g(w1+eps, w2, bias) - ccc) / eps
		dw2 := (cost_g(w1, w2+eps, bias) - ccc) / eps
		dbias := (cost_g(w1, w2, bias+eps) - ccc) / eps

		w1 -= rate * dw1
		w2 -= rate * dw2
		bias -= rate * dbias
	}

	fmt.Printf("cost =  %f, w1 =  %f, w2 = %f, bias = %f\n", cost_g(w1, w2, bias), w1, w2, bias)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			fmt.Printf("%d | %d => %f\n", i, j, sigmodf(float64(i)*w1+float64(j)*w2+bias))
		}
	}
}
