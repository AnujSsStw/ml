package main

import (
	"fmt"
	"math/rand"
	"time"
)

var train_data = [][2]float64{
	{0, 1},
	{1, 0},
	// {2, 4},
	// {3, 6},
	// {4, 8},
}

// 1 000 000 000 000 -> GPT4
// 1 -> ME
// y = x * w-> parameter which we have to find

// also called lost fn
func cost(w float64, b float64) float64 {
	var result float64 = 0
	for i := 0; i < len(train_data); i++ {
		x := train_data[i][0]
		// def. of single arti. neural which is
		// input = x1, x2, x3, ...
		// weight = w1, w2, w3, ...
		// y = x1*w1 + x2*w2 + ... + b
		// b -> bias

		// effectively think it as a single artif. neuron with a signle input and with single connection
		y := x*w + b
		diff := y - train_data[i][1]

		result += diff * diff
		// fmt.Printf("actual: %f, expected: %f\n", y, train_data[i][1])
	}

	result /= float64(len(train_data))
	return result
}

func main() {
	// rand seed
	s1 := rand.NewSource(time.Now().UnixNano())
	r := rand.New(s1)

	// _ = r
	w := r.Float64() * 10.0
	// w := float64(-10)
	// simply with derative the w flucate too much that's why we introduce learning rate
	rate := 1e-3
	eps := 1e-3
	var bias float64 = r.Float64() * 5.0

	for i := 0; i < 10000; i++ {
		ccc := cost(w, bias)
		dcost := (cost(w+eps, bias) - ccc) / eps
		dbias := (cost(w, bias+eps) - ccc) / eps
		w -= rate * dcost
		bias -= rate * dbias
		fmt.Printf("cost =  %f, w =  %f, bias = %f\n", cost(w, bias), w, bias)
	}

	fmt.Println("----------------------------")
	fmt.Printf("w: %f, bias: %f\n", w, bias)
	for i := 0; i < 2; i++ {
		fmt.Printf("%d => %f\n", i, w*train_data[i][0]+bias)
	}
}
