package oam

type TraitDef struct {
	ApiVersion string    `json:"apiVersion"`
	Kind       string    `json:"kind"`
	Metadata   Metadata  `json:"metadata"`
	Spec       TraitSpec `json:"spec"`
}

type TraitSpec struct {
	AppliesToWorkloads []string      `json:"appliesToWorkloads"`
	ConflictsWith      []string      `json:"conflictsWith"`
	DefinitionRef      DefinitionRef `json:"definitionRef"`
}

/*
* IMPORTANT: the thing to remember about traits is: some apply to nodes, some apply to workloads, some apply to both
* 			Some just have to be checked for presence, others have to be checked for values
* 			Anything that has to be checked for presence is done here, other stuff has to be done by the QoE calculator
 */

type TraitType string

type TraitImpl interface {
	/*CheckNodeTrait(node NodeSummary) bool
	CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool
	Apply(cdef ComponentDef)*/
}

var RequiredTraitProperty string = "Required"

func CheckNodeTrait(node NodeSummary, trait Trait) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == trait.Type {
			return true
		}
	}
	return false
}

func CheckWorkloadTypeTrait(traitDef TraitDef, wl WorkloadTypeDescriptor, trait Trait) bool {
	//hitMap := make(map[string]bool)
	//for _, node := range nodes {
	//for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
	if traitDef.Metadata.Name == trait.Type {
		for _, wlApplies := range traitDef.Spec.AppliesToWorkloads {
			if wlApplies == wl.Type {
				return true
				//hitMap[node.Name] = true
			}
		}
	}
	//}
	//}
	return false
}

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var NodeNetworkEncryptionTraitType TraitType = "NodeNetworkEncryption"

type NetworkEncryptionTrait struct {
	Required bool
}

/*func (t NetworkEncryptionTrait) Apply(cdef ComponentDef) {
	//TODO in Warrens
}*/

/*func (t NetworkEncryptionTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(NodeNetworkEncryptionTraitType) {
			return true
		}
	}
	return false
}

func (t NetworkEncryptionTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	wlTypes := GetSwirlySupportedWorkloadTypes()
	wlDef, found := wlTypes[wl.Type]
	if found {
		_, found := wlDef.Metadata.Labels[MetaSecureRuntimeLabel]
		return found
	}
	//Technically, an error
	return false
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var SecureRuntimeTraitType TraitType = "SecureRuntime"

type SecureRuntimeTrait struct {
	Required bool
}

/*func (t SecureRuntimeTrait) Apply(cdef ComponentDef) {

}

func (t SecureRuntimeTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(SecureRuntimeTraitType) {
			return true
		}
	}
	return false
}

func (t SecureRuntimeTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	wlTypes := GetSwirlySupportedWorkloadTypes()
	wlDef, found := wlTypes[wl.Type]
	if found {
		_, found := wlDef.Metadata.Labels[MetaSecureRuntimeLabel]
		return found
	}
	//Technically, an error
	return false
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var SecureEnclaveTraitType TraitType = "SecureEnclave"

type SecureEnclaveTrait struct {
	Required bool
}

/*func (t SecureEnclaveTrait) Apply(cdef ComponentDef) {
	//TODO in supported feather runtimes
}

func (t SecureEnclaveTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(SecureEnclaveTraitType) {
			return true
		}
	}
	return false
}

func (t SecureEnclaveTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var AttestationTraitType TraitType = "Attestation"

type AttestationTrait struct {
	Required bool
}

/*func (t AttestationTrait) Apply(cdef ComponentDef) {
	//TODO in TrustEdge
}

func (t AttestationTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(AttestationTraitType) {
			return true
		}
	}
	return false
}

func (t AttestationTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	wlTypes := GetSwirlySupportedWorkloadTypes()
	wlDef, found := wlTypes[wl.Type]
	if found {
		_, found := wlDef.Metadata.Labels[MetaAttestedSoftwareLabel]
		return found
	}
	return false
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var SecureDataTraitType TraitType = "SecureData"

type SecureDataTrait struct {
	Required bool
}

/*func (t SecureDataTrait) Apply(cdef ComponentDef) {
	//TODO in ??
}

func (t SecureDataTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(SecureDataTraitType) {
			return true
		}
	}
	return false
}

func (t SecureDataTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var EnergyEfficiencyTraitType TraitType = "EnergyEfficiency"

type EnergyEfficiencyTrait struct {
	Required bool
}

/*func (t EnergyEfficiencyTrait) Apply(cdef ComponentDef) {
	//TODO in ??
}

// Not applicable to node, so always true
func (t EnergyEfficiencyTrait) CheckNodeTrait(node NodeSummary) bool {
	return true
}

func (t EnergyEfficiencyTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	//Might be applicable to future workload types
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var GreenEnergyTraitType TraitType = "GreenEnergy"

type GreenEnergyTrait struct {
	Required bool
}

/*func (t GreenEnergyTrait) Apply(cdef ComponentDef) {
	//TODO in ??
}

func (t GreenEnergyTrait) CheckNodeTrait(node NodeSummary) bool {
	for _, traitDef := range node.NodeCaps.Caps.SupportedTraits {
		if traitDef.Metadata.Name == string(GreenEnergyTraitType) {
			return true
		}
	}
	return false
}

func (t GreenEnergyTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var SoftDistanceLimitTraitType TraitType = "discover.flocky.oam.SoftDistanceLimit"

type SoftDistanceLimitTrait struct {
	Max int
}

/*func (t SoftDistanceLimitTrait) Apply(cdef ComponentDef) {
	//Orchestrator thing, do not apply
}

// Any node has a QoE and distance, so yes
func (t SoftDistanceLimitTrait) CheckNodeTrait(node NodeSummary) bool {
	return true
}

// Not applicable to workload types
func (t SoftDistanceLimitTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

var MinimalResourcesTraitType TraitType = "MinimalResources"

type MinimalResourcesTrait struct {
	//Resources      v1.ResourceList
	//ResourceLimits v1.ResourceList
	//ResourceUse    v1.ResourceList
	//Required bool
}

/*func (t MinimalResourcesTrait) Apply(cdef ComponentDef) {
	//Applied by runtimes
}

// Any node can regulate resource limits, so yes
func (t MinimalResourcesTrait) CheckNodeTrait(node NodeSummary) bool {
	return true
}

// Not applicable to workload types, components have resource use instead
func (t MinimalResourcesTrait) CheckWorkloadTypeTrait(wl WorkloadTypeDescriptor) bool {
	return true
}*/

/*
---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
*/

func TraitFromComponentInfo(data Trait) TraitImpl {
	switch data.Type {
	/*case string(NodeNetworkEncryptionTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return NetworkEncryptionTrait{
			Required: required,
		}
	case string(AttestationTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return AttestationTrait{
			Required: required,
		}
	case string(SecureRuntimeTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return SecureRuntimeTrait{
			Required: required,
		}
	case string(SecureEnclaveTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return SecureEnclaveTrait{
			Required: required,
		}
	case string(SecureDataTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return SecureDataTrait{
			Required: required,
		}
	case string(EnergyEfficiencyTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return EnergyEfficiencyTrait{
			Required: required,
		}
	case string(GreenEnergyTraitType):
		required, _ := data.Properties[RequiredTraitProperty].(bool)
		return GreenEnergyTrait{
			Required: required,
		}*/
	case string(SoftDistanceLimitTraitType):
		min, _ := data.Properties["Max"].(int)
		//qoe, _ := strconv.ParseInt(min, 10, 32)
		return SoftDistanceLimitTrait{
			Max: int(min),
		}
	case string(MinimalResourcesTraitType):
		//required, _ := data.Properties["Required"].(bool)
		return MinimalResourcesTrait{
			//Required: required,
		}
	default:
		return MinimalResourcesTrait{}
	}
}

func ComponentTraitInfoFromTrait(obj interface{}) Trait {
	trait := Trait{}
	trait.Properties = Properties{}
	switch obj := obj.(type) {
	case MinimalResourcesTrait:
		//rlTrait, _ := obj.(ResourceLimitsTrait)
		trait.Type = string(MinimalResourcesTraitType)
		//trait.Properties["Required"] = obj.Required
		//trait.Properties["ResourceLimits"] = obj.ResourceLimits
		//trait.Properties["ResourceUse"] = obj.ResourceUse
	case SoftDistanceLimitTrait:
		//rlTrait, _ := obj.(ResourceLimitsTrait)
		trait.Type = string(SoftDistanceLimitTraitType)
		trait.Properties["Max"] = obj.Max
	case SecureRuntimeTrait:
		trait.Type = string(SecureRuntimeTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case AttestationTrait:
		trait.Type = string(AttestationTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case SecureEnclaveTrait:
		trait.Type = string(SecureEnclaveTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case SecureDataTrait:
		trait.Type = string(SecureDataTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case NetworkEncryptionTrait:
		trait.Type = string(NodeNetworkEncryptionTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case EnergyEfficiencyTrait:
		trait.Type = string(EnergyEfficiencyTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	case GreenEnergyTrait:
		trait.Type = string(GreenEnergyTraitType)
		trait.Properties[RequiredTraitProperty] = obj.Required
	}
	return trait
}

func IsRequiredTrait(trait Trait) bool {
	required, propExists := trait.Properties[RequiredTraitProperty]
	if propExists && required.(bool) {
		return true
	} else {
		return false
	}
}

func CreateTraitDef(kind string, workloads []WorkloadDef) TraitDef {
	workloadNames := []string{}
	for _, wl := range workloads {
		workloadNames = append(workloadNames, wl.Metadata.Name)
	}
	return TraitDef{
		Kind: "TraitDefinition",
		Metadata: Metadata{
			Name: kind,
		},
		Spec: TraitSpec{
			DefinitionRef: DefinitionRef{
				Name: kind,
			},
			AppliesToWorkloads: workloadNames,
		},
	}
}
