package oam

import (
	"oamswirly/common/hwresources"
	"time"

	v1 "k8s.io/api/core/v1"
)

type FlockyService string

var DiscoService FlockyService = "discover.flocky.oam.DiscoService"
var RepoService FlockyService = "discover.flocky.oam.RepoService"
var SwirlyService FlockyService = "discover.flocky.oam.SwirlyService"
var DeploymentService FlockyService = "discover.flocky.oam.DeploymentService"

type CapsProvider struct {
	Name          string
	LocalEndpoint string
}

type ServiceProvider struct {
	Name          FlockyService
	LocalEndpoint string
}

type TraitHandler struct {
	Trait         TraitDef
	LocalEndpoint string
}

type NodeCaps struct {
	ApiVersion string          `json:"apiVersion"`
	Kind       string          `json:"kind"`
	Metadata   Metadata        `json:"metadata"`
	Caps       NodeCapsContent `json:"caps"`
}

type NodeCapsContent struct {
	Resources          hwresources.NodeResources `json:"resources"`
	SupportedTraits    []TraitDef                `json:"supportedTraits"`
	SupportedWorkloads []WorkloadDef             `json:"supportedWorkloads"`
	Time               time.Time                 `json:"time"`
}

type NodeApps struct {
	Applications []SubApplication `json:"applications"`
	Time         time.Time        `json:"time"`
}

// var ResourceCPUEquivalent v1.ResourceName = "CPUEquivalent"
var ResourceNetwork v1.ResourceName = "network"

type NodeSummary struct {
	ApiVersion string   `json:"apiVersion"`
	Kind       string   `json:"kind"`
	Name       string   `json:"name"`
	NetInfo    NetInfo  `json:"netInfo"`
	Metadata   Metadata `json:"metadata"`
	NodeCaps   NodeCaps `json:"capabilities"`
	NodeApps   NodeApps `json:"applications"`
}

type NetInfo struct {
	PrimaryAddress string  `json:"primaryAddress"`
	Latency        float32 `json:"latency"`
}

func MergeCapsContent(target *NodeCapsContent, source NodeCapsContent) {
	/*for typ, res := range source.Resources {
		_, exists := target.Resources[typ]
		if !exists {
			target.Resources[typ] = res
		}
	}*/
	/*if source.Resources.CPUStats != nil {
		target.Resources = source.Resources
	}*/
	for rstype, content := range source.Resources.HwStats {
		target.Resources.HwStats[rstype] = content
	}

	for _, trait := range source.SupportedTraits {
		found := false
		for _, exTrait := range target.SupportedTraits {
			if exTrait.Metadata.Name == trait.Metadata.Name {
				found = true
			}
		}
		if !found {
			target.SupportedTraits = append(target.SupportedTraits, trait)
		}
	}

	for _, wl := range source.SupportedWorkloads {
		found := false
		for _, exWl := range target.SupportedWorkloads {
			if exWl.Metadata.Name == wl.Metadata.Name {
				found = true
			}
		}
		if !found {
			target.SupportedWorkloads = append(target.SupportedWorkloads, wl)
		}
	}
}

func TraitDefArrayDiff(list []TraitDef, excl []TraitDef) []TraitDef {
	diff := []TraitDef{}
	for _, item := range list {
		notInList := true
		for _, existing := range excl {
			if item.Metadata.Name == existing.Metadata.Name {
				notInList = false
			}
		}
		if notInList {
			diff = append(diff, item)
		}
	}
	return diff
}

func TraitArrayDiff(list []Trait, excl []Trait) []Trait {
	diff := []Trait{}
	for _, item := range list {
		notInList := true
		for _, existing := range excl {
			if item.Type == existing.Type {
				notInList = false
			}
		}
		if notInList {
			diff = append(diff, item)
		}
	}
	return diff
}

func TraitsEqual(traits1 []Trait, traits2 []Trait) bool {
	return len(TraitArrayDiff(traits1, traits2)) == 0
}

func WorkloadArrayDiff(list []WorkloadDef, excl []WorkloadDef) []WorkloadDef {
	diff := []WorkloadDef{}
	for _, item := range list {
		notInList := true
		for _, existing := range excl {
			if item.Metadata.Name == existing.Metadata.Name {
				notInList = false
			}
		}
		if notInList {
			diff = append(diff, item)
		}
	}
	return diff
}

func ApplicationArrayDiff(list []SubApplication, excl []SubApplication) []SubApplication {
	diff := []SubApplication{}
	for _, app := range list {
		notInList := true
		for _, existingApp := range excl {
			if app.Metadata.Name == existingApp.Metadata.Name {
				notInList = false
			}
		}
		if notInList {
			diff = append(diff, app)
		}
	}
	return diff
}
