//go:build !windows

package main

import (
	"os"
	"os/signal"
	"syscall"
	"test/mmap/config"
	"test/mmap/gossiping"
	"test/mmap/sharedmem"
	"test/mmap/utils"

	//gossipConfig "test/mmap/gossipConfig"

	log "github.com/sirupsen/logrus"
)

func main() {
	//-------------------------------------------------------
	// Load configs
	//-------------------------------------------------------
	/*if err := config.LoadConfig("config.json"); err != nil {
		log.Fatalf("Load config error %v", err.Error())
	}*/
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultConfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}
	if err := config.LoadConfig(cfgFile); err != nil {
		log.Fatalf("Load config error %v", err.Error())
	}

	var prefix string
	if len(argsWithoutProg) > 1 {
		prefix = argsWithoutProg[1]
	}
	if prefix == "" {
		prefix = "f1"
	}

	node_count, err := utils.ExtractAfterF(prefix)
	if err != nil {
		panic(err)
	}
	log.Infof("Node count is %d", node_count) // → 75

	gossiping.InitGossiping()

	sigc := make(chan os.Signal, 1)
	signal.Notify(sigc, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigc
		log.Infof("Signal %v detected, shutting down", sig.String())
		sharedmem.Cleanup()
		gossiping.Cleanup()
		os.Exit(0)
	}()

	defer sharedmem.Cleanup()

	weightCh := make(chan sharedmem.WeightUpdate, 1)      // Buffered channel to avoid blocking
	writeWeightCh := make(chan sharedmem.WeightUpdate, 1) // Buffered channel to avoid blocking
	//writeUpdateCh := make(chan [][]float32, 1)            // Buffered channel to avoid blocking
	nCh := make(chan int, 1)

	sharedmem.Init(prefix, weightCh, writeWeightCh, nCh)
	//fakeGossip(weightCh, writeWeightCh)
	gossiping.StartGossip(weightCh, writeWeightCh, nCh)
}

/*func fakeGossip(weightCh chan [][]float32, writeWeightCh chan sharedmem.WeightUpdate) {
	for {
		weights := <-weightCh
		update := createMatrixWithValue(weights, 0.1)

		wupdate := sharedmem.WeightUpdate{NodeID: "test1", BaseWeights: weights, Update: update}
		sendUpdate(wupdate, writeWeightCh)

		wupdate = sharedmem.WeightUpdate{NodeID: "test2", BaseWeights: weights, Update: update}
		sendUpdate(wupdate, writeWeightCh)
	}
}

func sendUpdate(wupdate sharedmem.WeightUpdate, writeWeightCh chan sharedmem.WeightUpdate) {
	sent := false
	for !sent {
		select {
		case writeWeightCh <- wupdate:
			sent = true
		default:
		}
	}
}

func addToMatrix(matrix [][]float32, n float32) [][]float32 {
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			matrix[i][j] += n
		}
	}
	return matrix
}

func createMatrixWithValue(base [][]float32, n float32) [][]float32 {
	matrix := [][]float32{}
	for i := range base {
		arr := []float32{}
		for j := 0; j < len(base[i]); j++ {
			arr = append(arr, n)
		}
		matrix = append(matrix, arr)
	}
	return matrix
}*/
