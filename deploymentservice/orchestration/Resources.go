package orchestration

import (
	"oamswirly/common/hwresources"

	v1 "k8s.io/api/core/v1"
)

/*type ResourceUpdate struct {
	NodeId         string
	Resources      map[Resource]int
	TotalResources map[Resource]int
}*/

func resourcesFree(resources map[v1.ResourceName]int) bool {
	//log.Info("Check resources")

	/*if cores == 0 {
		cores = getCores()
	}*/
	resourcesFree := true
	used, total := hwresources.GetSimplifiedResources()
	for resource, amount := range resources {
		resUsed, measured := used[resource]
		resTotal, _ := total[resource]
		resMax := resTotal

		//log.Infof("Resource %s required %d max %d", resource, resUsed+amount, resMax)
		if measured && (resUsed+amount) > resMax {
			resourcesFree = false
		}
	}

	return resourcesFree
}

/*func getResources() (map[Resource]int, map[Resource]int) {
	resources := make(map[Resource]int)

	resources[CPUNanos] = hwresources.GetCPU()
	resources[Memory] = hwresources.GetFreeMemory()

	totalResources := make(map[Resource]int)

	totalResources[CPUShares] = 1000 * hwresources.GetCores()
	totalResources[Memory] = hwresources.GetTotalMemory()
	return resources, totalResources
}*/
