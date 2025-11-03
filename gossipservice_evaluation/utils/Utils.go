package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"test/mmap/sharedmem"
	"time"

	log "github.com/sirupsen/logrus"
)

// Helper function to log weight details
func LogWeightDetails(weights [][]float32, context string, metadata sharedmem.WeightMetadata) {
	if weights == nil {
		log.Infof("[%s] Weights are nil", context)
		return
	}
	// Check if originalShapes are available and non-empty for the first array
	if metadata.Shapes == nil || len(metadata.Shapes) == 0 {
		log.Warnf("[%s] Original shapes metadata is nil or empty. Cannot log detailed shape for the first array. Falling back to basic info.", context)
		// Fallback to basic logging for the first array if shapes are not available
		firstLayerData := weights[0] // Safe due to len(weights) == 0 check above
		if firstLayerData != nil && len(firstLayerData) > 0 {
			log.Infof("[%s] First Layer (fallback, no shape metadata): Flattened length (%d), First element: %f, Last element: %f",
				context, len(firstLayerData), firstLayerData[0], firstLayerData[len(firstLayerData)-1])
		} else {
			log.Infof("[%s] First Layer (fallback, no shape metadata): Layer is nil or empty.", context)
		}
		return
	}
	firstLayerData := weights[0]
	firstActualShape := metadata.Shapes[0]

	if firstLayerData != nil && len(firstLayerData) > 0 {
		log.Infof("[%s] First Layer: Original Shape %v, Flattened length (%d), First element: %f, Last element: %f",
			context, firstActualShape, len(firstLayerData), firstLayerData[0], firstLayerData[len(firstLayerData)-1])
	} else {
		log.Infof("[%s] First Layer: Original Shape %v, Layer is nil or empty.", context, firstActualShape)
	}
}

func WriteWeightsToFile(weights [][]float32, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer f.Close()

	// Write each weight array on a new line
	for _, layer := range weights {
		fmt.Fprintf(f, "%v", layer)
	}
	return nil
}

func AvgWeights(old [][]float32, new [][]float32) [][]float32 {
	if len(old) != len(new) {
		panic("weight slices must have same length")
	}

	// Reuse the old array to avoid allocations
	for i := range old {
		if len(old[i]) != len(new[i]) {
			panic("weight arrays must have same length")
		}

		// SIMD-friendly sequential memory access
		for j := 0; j < len(old[i]); j += 4 {
			remaining := len(old[i]) - j
			if remaining >= 4 {
				// Process 4 elements at once
				old[i][j] = (old[i][j] + new[i][j]) * 0.5
				old[i][j+1] = (old[i][j+1] + new[i][j+1]) * 0.5
				old[i][j+2] = (old[i][j+2] + new[i][j+2]) * 0.5
				old[i][j+3] = (old[i][j+3] + new[i][j+3]) * 0.5
			} else {
				// Handle remaining elements
				for k := 0; k < remaining; k++ {
					old[i][j+k] = (old[i][j+k] + new[i][j+k]) * 0.5
				}
			}
		}
	}
	return old
}

func TakeAverage(num_items int, summed_matrix [][]float32) [][]float32 {
	if summed_matrix == nil {
		return nil
	}
	if num_items <= 0 {
		log.Errorf("take_average called with num_items = %d, which is invalid", num_items)
		return summed_matrix
	}

	result := make([][]float32, len(summed_matrix))
	for i := 0; i < len(summed_matrix); i++ {
		if summed_matrix[i] == nil {
			result[i] = nil // Preserve nil inner slices
			continue
		}
		result[i] = make([]float32, len(summed_matrix[i]))
		for j := 0; j < len(summed_matrix[i]); j++ {
			result[i][j] = summed_matrix[i][j] / float32(num_items)
		}
	}
	return result
}

func MatrixSum(a, b [][]float32) [][]float32 {
	if len(a) != len(b) {
		panic("matrix dimensions do not match")
	}

	result := make([][]float32, len(a))
	for i := range a {
		if len(a[i]) != len(b[i]) {
			panic("matrix row lengths do not match")
		}
		result[i] = make([]float32, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func MatrixDiff(a, b [][]float32) [][]float32 {
	if len(a) != len(b) {
		panic("matrix dimensions do not match")
	}

	result := make([][]float32, len(a))
	for i := range a {
		if len(a[i]) != len(b[i]) {
			panic("matrix row lengths do not match")
		}
		result[i] = make([]float32, len(a[i]))
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func AppendMetrics(filename string, nodeID string, iteration int, updates int, numNeighbours int, timestamp time.Time) error {
	// Open the file in append mode, create it if it doesn't exist
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open metrics file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Check if the file is new or empty to write headers
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to get file info: %w", err)
	}

	if info.Size() == 0 {
		headers := []string{"nodeID", "iteration", "updates", "num_neighbours", "timestamp"}
		if err := writer.Write(headers); err != nil {
			return fmt.Errorf("failed to write headers to metrics file: %w", err)
		}
	}

	// Prepare the data row
	row := []string{
		nodeID,
		strconv.Itoa(iteration),
		strconv.Itoa(updates),
		strconv.Itoa(numNeighbours),
		timestamp.Format(time.RFC3339Nano), // Using a standard timestamp format
	}

	// Write the new row
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write to metrics file: %w", err)
	}

	return nil
}

func ExtractAfterF(prefix string) (int, error) {
	// Compile once if in hot path
	re := regexp.MustCompile(`^f(\d+)`)
	parts := re.FindStringSubmatch(prefix)
	if len(parts) < 2 {
		return 0, fmt.Errorf("no number after 'f' in %q", prefix)
	}
	return strconv.Atoi(parts[1])
}

/*func readWeightsFromFile(filename string) ([][]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("open file: %v", err)
	}
	defer f.Close()

	var weights [][]float32
	scanner := bufio.NewScanner(f)

	// Increase scanner buffer size to 10MB
	const maxCapacity = 10 * 1024 * 1024
	buf := make([]byte, maxCapacity)
	scanner.Buffer(buf, maxCapacity)

	for scanner.Scan() {
		line := scanner.Text()
		// Remove 'Layer X:' prefix if present
		if idx := strings.Index(line, ":"); idx != -1 {
			line = line[idx+1:]
		}

		// Clean up the line
		line = strings.TrimSpace(line)
		line = strings.Trim(line, "[]")

		// Split on comma and space
		numStrings := strings.FieldsFunc(line, func(r rune) bool {
			return r == ',' || r == ' ' || r == '[' || r == ']'
		})

		// Convert strings to float32
		layerWeights := make([]float32, 0, len(numStrings))
		for _, numStr := range numStrings {
			numStr = strings.TrimSpace(numStr)
			if numStr == "" {
				continue
			}
			num, err := strconv.ParseFloat(numStr, 32)
			if err != nil {
				return nil, fmt.Errorf("parse float: %v in string: %q", err, numStr)
			}
			layerWeights = append(layerWeights, float32(num))
		}

		if len(layerWeights) > 0 {
			weights = append(weights, layerWeights)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanner error: %v", err)
	}

	return weights, nil
}*/
