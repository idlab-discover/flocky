package oam

import (
	"log"
	"oamswirly/common/hwresources"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

var ResourceAccelerator hwresources.ResourceType = "Accelerator"

type AcceleratorStats struct {
	TotalMemoryBytes             uint64
	UsedMemoryBytes              uint64
	Accelerators                 []Accelerator
	ConfidentialComputeAbilities uint32
}

type Accelerator struct {
	Model            string
	Architecture     nvml.DeviceArchitecture
	Brand            nvml.BrandType
	MemoryTotalBytes uint64
	MemoryUsedBytes  uint64
}

func GetAcceleratorStats() AcceleratorStats {
	gpusConfCapabilities := HasConfComputing()

	gpus := GetAccelerators()
	totMemory := uint64(0)
	usedMemory := uint64(0)
	for _, gpu := range gpus {
		totMemory += gpu.MemoryTotalBytes
		usedMemory += gpu.MemoryUsedBytes
	}

	return AcceleratorStats{
		TotalMemoryBytes:             totMemory,
		UsedMemoryBytes:              usedMemory,
		Accelerators:                 gpus,
		ConfidentialComputeAbilities: gpusConfCapabilities,
	}
}

func GetAccelerators() []Accelerator {
	accelerators := []Accelerator{}

	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to initialize NVML: %v", nvml.ErrorString(ret))
		return accelerators
	}
	defer func() {
		ret := nvml.Shutdown()
		if ret != nvml.SUCCESS {
			log.Fatalf("Unable to shutdown NVML: %v", nvml.ErrorString(ret))
		}
	}()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to get device count: %v", nvml.ErrorString(ret))
		return accelerators
	}

	for i := 0; i < count; i++ {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			log.Fatalf("Unable to get device at index %d: %v", i, nvml.ErrorString(ret))
		}

		name, _ := device.GetName()
		arch, _ := device.GetArchitecture()
		brand, _ := device.GetBrand()
		mem, _ := device.GetMemoryInfo()

		accelerators = append(accelerators, Accelerator{
			Model:            name,
			Architecture:     arch,
			Brand:            brand,
			MemoryTotalBytes: mem.Total,
			MemoryUsedBytes:  mem.Used,
		})
	}
	return accelerators
}

func HasAccelerators() bool {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to initialize NVML: %v", nvml.ErrorString(ret))
		return false
	}
	defer func() {
		ret := nvml.Shutdown()
		if ret != nvml.SUCCESS {
			log.Fatalf("Unable to shutdown NVML: %v", nvml.ErrorString(ret))
		}
	}()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to get device count: %v", nvml.ErrorString(ret))
		return false
	}
	return count > 0
}

func HasConfComputing() uint32 {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to initialize NVML: %v", nvml.ErrorString(ret))
	}

	confCapabilities, _ := nvml.SystemGetConfComputeCapabilities()
	gpusConfCapabilities := confCapabilities.GpusCaps

	ret = nvml.Shutdown()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to shutdown NVML: %v", nvml.ErrorString(ret))
	}
	return gpusConfCapabilities
}
