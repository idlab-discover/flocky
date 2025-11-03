package oam

import (
	v1 "k8s.io/api/core/v1"
)

//var MetaBaseComponentLabel string = "BaseComponent"

var MetaMinClients string = "ClientsMin"
var MetaMaxClients string = "ClientsMax"

var MetaScalability string = "ScalabilityIdx"

// var ResourceCPULoad v1.ResourceName = "CPULoad"
// var ResourceMemoryUse v1.ResourceName = "MemoryUse"
var ResourceEnergyUseTask v1.ResourceName = "EnergyUse"
var ResourcePowerUseRunning v1.ResourceName = "PowerUse"

//var MetaEnergy

type ComponentDef struct {
	ApiVersion string
	Kind       string
	Metadata   Metadata
	Spec       ComponentDefSpec
}

type ComponentDefSpec struct {
	Workload  WorkloadTypeDescriptor
	Schematic Schematic
}

type WorkloadTypeDescriptor struct {
	Type       string
	Definition WorkloadGVK
}

type WorkloadGVK struct {
	ApiVersion string
	Kind       string
}

type Schematic struct {
	Definition    v1.Container
	Params        map[string]string
	BaseComponent string
	BaseResources v1.ResourceList
	MaxResources  v1.ResourceList
}

/*func test() {
	test := ComponentDefSpec{
		Schematic: WorkloadGVK{},
	}
}*/
