package oam

type WorkloadDef struct {
	ApiVersion string          `json:"apiVersion"`
	Kind       string          `json:"kind"`
	Spec       WorkloadDefSpec `json:"spec"`
	Metadata   Metadata        `json:"metadata"`
}

type WorkloadDefSpec struct {
	DefinitionRef DefinitionRef `json:"definitionRef"`
}

type DefinitionRef struct {
	Name       string `json:"name"`
	ApiVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
}

type RuntimeLabel string

var MetaRuntimeLabel string = "Runtime"
var ContainerRuntime RuntimeLabel = "Container"
var UnikernelRuntime RuntimeLabel = "Unikernel"

var MetaSecureRuntimeLabel string = "SecureRuntime"
var MetaAttestedSoftwareLabel string = "AttestedSoftware"

//Meta label for attestation?
//Meta label for software security?

func GetContainerServiceType() WorkloadDef {
	labels := make(map[string]string)
	labels[MetaRuntimeLabel] = string(ContainerRuntime)
	return WorkloadDef{
		ApiVersion: "core.oam.dev/v1beta1",
		Kind:       "WorkloadDefinition",
		Metadata: Metadata{
			Name:   string(ContainerRuntime),
			Labels: labels,
		},
		Spec: WorkloadDefSpec{
			DefinitionRef: DefinitionRef{
				Name: "containerizedworkloads.core.oam.dev",
			},
		},
	}
}

func GetUnikernelServiceType() WorkloadDef {
	labels := make(map[string]string)
	labels[MetaRuntimeLabel] = string(UnikernelRuntime)
	labels[MetaSecureRuntimeLabel] = "true"
	return WorkloadDef{
		ApiVersion: "core.oam.dev/v1beta1",
		Kind:       "WorkloadDefinition",
		Metadata: Metadata{
			Name:   string(UnikernelRuntime),
			Labels: labels,
		},
		Spec: WorkloadDefSpec{
			DefinitionRef: DefinitionRef{
				Name: "containerizedworkloads.core.oam.dev",
			},
		},
	}
}

func GetSwirlySupportedWorkloadTypes() map[string]WorkloadDef {
	wlDefs := make(map[string]WorkloadDef)
	wlDefs[string(ContainerRuntime)] = GetContainerServiceType()
	wlDefs[string(UnikernelRuntime)] = GetUnikernelServiceType()

	return wlDefs
}

func GetSwirlySupportedWorkloadTypesAlt() []WorkloadDef {
	wlDefs := []WorkloadDef{GetContainerServiceType(), GetUnikernelServiceType()}

	return wlDefs
}
