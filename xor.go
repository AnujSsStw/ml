package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Xor struct {
	or_w1   float64
	or_w2   float64
	or_bias float64

	nand_w1   float64
	nand_w2   float64
	nand_bias float64

	and_w1   float64
	and_w2   float64
	and_bias float64
}

var train_data_xor = [][3]float64{
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
}

func sigmodfx(x float64) float64 {
	return float64(1) / (float64(1) + math.Exp(-x))
}

func rand_xor(r *rand.Rand) Xor {
	m := Xor{
		or_w1:     r.Float64(),
		or_w2:     r.Float64(),
		or_bias:   r.Float64(),
		nand_w1:   r.Float64(),
		nand_w2:   r.Float64(),
		nand_bias: r.Float64(),
		and_w1:    r.Float64(),
		and_w2:    r.Float64(),
		and_bias:  r.Float64(),
	}
	return m

}

func forward(m Xor, x1 float64, x2 float64) float64 {
	var x float64 = sigmodfx(m.or_w1*x1 + m.or_w2*x2 + m.or_bias)
	var y float64 = sigmodfx(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_bias)
	result := sigmodfx(m.and_w1*x + m.and_w2*y + m.and_bias)
	return result
}

func xor_cost(m Xor) float64 {
	var result float64 = 0
	for i := 0; i < len(train_data_xor); i++ {
		x1 := train_data_xor[i][0]
		x2 := train_data_xor[i][1]
		y := forward(m, x1, x2)
		diff := y - train_data_xor[i][2]

		result += diff * diff
		// fmt.Printf("actual: %f, expected: %f\n", y, train_data_xor[i][1])
	}

	result /= float64(len(train_data_xor))
	return result
}

func finite_derative(m Xor, eps float64) Xor {
	var dummy Xor
	ccc := xor_cost(m)
	var saved float64

	saved = m.or_w1
	m.or_w1 += eps
	dummy.or_w1 = (xor_cost(m) - ccc) / eps
	m.or_w1 = saved

	saved = m.or_w2
	m.or_w2 += eps
	dummy.or_w1 = (xor_cost(m) - ccc) / eps
	m.or_w2 = saved

	saved = m.or_bias
	m.or_bias += eps
	dummy.or_bias = (xor_cost(m) - ccc) / eps
	m.or_bias = saved

	saved = m.nand_w1
	m.nand_w1 += eps
	dummy.nand_w1 = (xor_cost(m) - ccc) / eps
	m.nand_w1 = saved

	saved = m.nand_w2
	m.nand_w2 += eps
	dummy.nand_w2 = (xor_cost(m) - ccc) / eps
	m.nand_w2 = saved

	saved = m.nand_bias
	m.nand_bias += eps
	dummy.nand_bias = (xor_cost(m) - ccc) / eps
	m.nand_bias = saved

	saved = m.and_w1
	m.and_w1 += eps
	dummy.and_w1 = (xor_cost(m) - ccc) / eps
	m.and_w1 = saved

	saved = m.and_w2
	m.and_w2 += eps
	dummy.and_w2 = (xor_cost(m) - ccc) / eps
	m.and_w2 = saved

	saved = m.and_bias
	m.and_bias += eps
	dummy.and_bias = (xor_cost(m) - ccc) / eps
	m.and_bias = saved

	return dummy
}

func train(m Xor, dummy Xor, rate float64) Xor {
	m.or_w1 -= rate * dummy.or_w1
	m.or_w2 -= rate * dummy.or_w2
	m.or_bias -= rate * dummy.or_bias
	m.nand_w1 -= rate * dummy.nand_w1
	m.nand_w2 -= rate * dummy.nand_w2
	m.nand_bias -= rate * dummy.nand_bias
	m.and_w1 -= rate * dummy.and_w1
	m.and_w2 -= rate * dummy.and_w2
	m.and_bias -= rate * dummy.and_bias

	return m
}

func main() {
	// (x|y) & ~(x&y)
	s1 := rand.NewSource(time.Now().UnixNano())
	r := rand.New(s1)

	m := rand_xor(r)
	rate := 1e-1
	eps := 1e-1
	// fmt.Println(m)

	for i := 0; i < 5000*500; i++ {
		dummy := finite_derative(m, eps)
		m = train(m, dummy, rate)
	}

	fmt.Printf("cost =  %f\n", xor_cost(m))
	fmt.Println(m)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			fmt.Printf("%d ^ %d => %f\n", i, j, forward(m, float64(i), float64(j)))
		}
	}

	fmt.Println("-------------------------------------------\n")
	fmt.Println("or tt:")
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			fmt.Printf("%d | %d => %f\n", i, j, sigmodfx(float64(i)*m.or_w1+float64(j)*m.or_w2+m.or_bias))
		}
	}
}
