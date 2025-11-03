package ws

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"oamswirly/common/hwresources"
	"oamswirly/common/oam"
	"oamswirly/dummyservices/config"

	v1 "k8s.io/api/core/v1"

	//"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

var simulatedPods map[string]v1.Pod
var simulateApps map[string]oam.SubApplication

func init() {
	simulatedPods = make(map[string]v1.Pod)
	simulateApps = make(map[string]oam.SubApplication)
}

// GET /ping
func GetPods(w http.ResponseWriter, r *http.Request) {
	//log.Info("GetPods")

	pods := []oam.SubApplication{}
	for _, pod := range simulateApps {
		pods = append(pods, pod)
	}

	jsonBytes, _ := json.Marshal(pods)
	w.Write(jsonBytes)
	//w.WriteHeader(200)
}

func DeployPod(w http.ResponseWriter, r *http.Request) {
	//log.Info("DeployPod")

	decoder := json.NewDecoder(r.Body)
	pod := &v1.Pod{}
	err := decoder.Decode(pod)
	if err != nil {
		log.Errorf("JSON decode error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	//log.Infof("Deploying %s", pod.Name)

	simulatedPods[pod.Name] = *pod
}

func DeletePod(w http.ResponseWriter, r *http.Request) {
	//log.Info("DeletePod")

	decoder := json.NewDecoder(r.Body)
	pod := &v1.Pod{}
	err := decoder.Decode(pod)
	if err != nil {
		log.Errorf("JSON decode error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	//log.Infof("Deleting %s", pod.Name)

	delete(simulatedPods, pod.Name)
}

func DeployOAMPod(w http.ResponseWriter, r *http.Request) {
	//log.Info("DeployOAMPod")

	decoder := json.NewDecoder(r.Body)
	pod := &oam.SubApplication{}
	err := decoder.Decode(pod)
	if err != nil {
		log.Errorf("JSON decode error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	//log.Infof("Deploying %s", pod.Metadata.Name)

	simulateApps[pod.Metadata.Name] = *pod
}

func DeleteOAMPod(w http.ResponseWriter, r *http.Request) {
	//log.Info("DeleteOAMPod")

	decoder := json.NewDecoder(r.Body)
	pod := &oam.SubApplication{}
	err := decoder.Decode(pod)
	if err != nil {
		log.Errorf("JSON decode error %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
	}
	//log.Infof("Deleting %s", pod.Metadata.Name)

	delete(simulateApps, pod.Metadata.Name)
}

func getPodSpec(name string) (*v1.Pod, error) {
	podBytes, err := ioutil.ReadFile(fmt.Sprintf("services/%s.json", name))
	pod := &(v1.Pod{})

	if err != nil {
		return nil, err
	}

	json.Unmarshal(podBytes, pod)
	return pod, nil
}

func parsePodSpec(component oam.Component, def oam.ComponentDef) *v1.Pod {
	pod := v1.Pod{}
	pod.Spec.Containers = []v1.Container{
		def.Spec.Schematic.Definition,
	}
	wltypes := oam.GetSwirlySupportedWorkloadTypes()
	specificType := wltypes[def.Spec.Workload.Type]
	runtime, found := specificType.Metadata.Labels[oam.MetaRuntimeLabel]
	//TODO: don't need to do anything with this for now, but it's here if needed for traits
	if found {
		switch runtime {
		case string(oam.UnikernelRuntime):
			break
		default:
			break
		}
	}
	//TODO: any scopes (e.g. storagescope for volumes, sidecar scope for stuff to be combined
	return &pod
}

func createTraitDef(kind string) oam.TraitDef {
	return oam.TraitDef{
		Kind: "TraitDefinition",
		Metadata: oam.Metadata{
			Name: kind,
		},
		Spec: oam.TraitSpec{
			DefinitionRef: oam.DefinitionRef{
				Name: kind,
			},
		},
	}
}

func ListFeatherCapabilities(w http.ResponseWriter, r *http.Request) {
	//log.Info("ListFeatherCapabilities")
	caps := oam.NodeCapsContent{}
	//resources
	if config.Cfg.SimCores == 0 {
		//log.Info("Getting hardware resources")
		caps.Resources = hwresources.GetResources()
	} else {
		//log.Info("Simulating hardware resources")
		caps.Resources = GetSimulatedResources()
	}

	//traits
	caps.SupportedTraits = []oam.TraitDef{}
	if config.Cfg.SimAttestationTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, createTraitDef(string(oam.AttestationTraitType)))
	}
	if config.Cfg.SimGreenEnergyTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, createTraitDef(string(oam.GreenEnergyTraitType)))
	}
	if config.Cfg.SimSecureEnclaveTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, createTraitDef(string(oam.SecureEnclaveTraitType)))
	}
	if config.Cfg.SimSecureRuntimeTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, createTraitDef(string(oam.SecureRuntimeTraitType)))
	}

	//workloads
	allWorkloads := oam.GetSwirlySupportedWorkloadTypes()
	caps.SupportedWorkloads = []oam.WorkloadDef{allWorkloads[string(oam.ContainerRuntime)]}

	if config.Cfg.SimSecureRuntimeTrait {
		caps.SupportedWorkloads = append(caps.SupportedWorkloads, allWorkloads[string(oam.UnikernelRuntime)])
	}

	json, err := json.Marshal(caps)
	if err != nil {
		log.Errorf("ListFeatherCapabilities JSON error %v", err)
		return
	}
	_, err = w.Write(json)
	if err != nil {
		log.Errorf("ListFeatherCapabilities error %v", err)
	}
}

func GetSimulatedResources() hwresources.NodeResources {
	//panic("unimplemented")
	baseResources := hwresources.GetResources()

	totalNanocores := uint64(config.Cfg.SimCores * 1000000000)
	totalMemory := uint64(config.Cfg.SimMemory * 1024 * 1024)
	//log.Infof("Simulating %d nanocores %d memory bytes", totalNanocores, totalMemory)

	usedNanocores := uint64(0)
	freeMemory := totalMemory
	usedMemory := uint64(0)
	for _, app := range simulateApps {
		for _, cDef := range app.Spec.ConcreteComponents {
			rReqs := cDef.Spec.Schematic.Definition.Resources.Requests
			cpuReq, _ := rReqs.Cpu().AsInt64()
			usedNanocores += uint64(cpuReq * 1000000000)

			memReq, _ := rReqs.Memory().AsInt64()
			freeMemory -= uint64(memReq)
			usedMemory += uint64(memReq)
		}
	}

	baseResources.HwStats[hwresources.ResourceCPU] = hwresources.CPUStats{
		CPUStats: stats.CPUStats{
			UsageNanoCores: &usedNanocores,
		},
		CPUNanoCores:  &totalNanocores,
		CPUEquivalent: 1,
	}
	baseResources.HwStats[hwresources.ResourceMemory] = &hwresources.MemoryStats{
		MemoryStats: stats.MemoryStats{
			AvailableBytes: &freeMemory,
			UsageBytes:     &usedMemory,
		},
		TotalBytes: &totalMemory,
	}

	return baseResources
}

func NodesDiscovered(w http.ResponseWriter, r *http.Request) {
	//log.Info("NodesDiscovered")
}

func ListWarrensCapabilities(w http.ResponseWriter, r *http.Request) {
	//log.Info("ListWarrensCapabilities")
	caps := oam.NodeCapsContent{}

	caps.SupportedTraits = []oam.TraitDef{}
	if config.Cfg.SimNetworkEncryptionTrait {
		caps.SupportedTraits = append(caps.SupportedTraits, createTraitDef(string(oam.NodeNetworkEncryptionTraitType)))
	}

	json, err := json.Marshal(caps)
	if err != nil {
		log.Errorf("ListWarrensCapabilities JSON error %v", err)
		return
	}
	//log.Infof("Sending %s", json)
	_, err = w.Write(json)
	if err != nil {
		log.Errorf("ListWarrensCapabilities error %v", err)
	}
}

//var reInsideWhtsp = regexp.MustCompile(`\s+`)
