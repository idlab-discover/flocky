package local

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"time"

	log "github.com/sirupsen/logrus"
)

var capsProviders map[string]string
var applicationProviders map[string]string
var traitHandlers map[string]oam.TraitHandler
var flockyServices map[oam.FlockyService]string

/*type CapsScanner interface {
	Scan() (oam.NodeCapsContent, error)
}*/

func Init() {
	capsProviders = make(map[string]string)
	applicationProviders = make(map[string]string)
	traitHandlers = make(map[string]oam.TraitHandler)
	flockyServices = make(map[oam.FlockyService]string)
	//Default Flocky services can be added with their standard ports
	//Note that services are NOT the same as capabilityproviders, since services might not provide capabilities, and capabilityproviders might not provide an actual service
	flockyServices[oam.DiscoService] = "31000"
	flockyServices[oam.RepoService] = "32000"
	//Although Swirly and Deploy are core services, they're not necessarily used so we'll leave it to those services to register themselves
}

var LocalCaps oam.NodeCapsContent
var LocalApps oam.NodeApps

func SyncLocalCapabilities(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				LocalApps, _ = GetApplicationsSummary()
				LocalCaps, _ = GetCapabilitiesSummary()
				time.Sleep(time.Second)
			}
		}
	}()
}

func GetCapabilitiesSummary() (oam.NodeCapsContent, error) {
	totalCaps := &oam.NodeCapsContent{
		Time:               time.Now(),
		SupportedTraits:    []oam.TraitDef{},
		SupportedWorkloads: []oam.WorkloadDef{},
	}

	for _, port := range capsProviders {
		response, err := http.Get(fmt.Sprintf("http://localhost:%s/listCapabilities", port))
		providerCaps := oam.NodeCapsContent{}
		if err != nil {
			log.Errorf("GetCapabilitiesSummary GET error on port %s %v", port, err)
			return *totalCaps, err
		}

		err = json.NewDecoder(response.Body).Decode(&providerCaps)
		if err != nil {
			log.Errorf("GetCapabilitiesSummary JSON error %v", err)
			return *totalCaps, err
		}
		oam.MergeCapsContent(totalCaps, providerCaps)

		response.Body.Close()
	}
	return *totalCaps, nil
}

func GetApplicationsSummary() (oam.NodeApps, error) {
	totalApps := oam.NodeApps{
		Time: time.Now(),
	}
	for _, port := range applicationProviders {
		response, err := http.Get(fmt.Sprintf("http://localhost:%s/getDeployedComponents", port))
		providerApps := []oam.SubApplication{}
		if err != nil {
			log.Errorf("GetApplicationsSummary GET error on port %s %v", port, err)
			return totalApps, err
		}

		err = json.NewDecoder(response.Body).Decode(&providerApps)
		if err != nil {
			log.Errorf("GetApplicationsSummary JSON error %v", err)
			return totalApps, err
		}
		totalApps.Applications = append(totalApps.Applications, providerApps...)

		response.Body.Close()
	}
	return totalApps, nil
}

func RegisterCapabilitiesProvider(name string, endpoint string) {
	capsProviders[name] = endpoint
}

func RegisterAppsProvider(name string, endpoint string) {
	applicationProviders[name] = endpoint
}

func RegisterTraitHandler(handler oam.TraitHandler) {
	traitHandlers[handler.Trait.Metadata.Name] = handler
}

func GetTraitHandler(trait string) oam.TraitHandler {
	endpoint, found := traitHandlers[trait]
	if found {
		return endpoint
	} else {
		return oam.TraitHandler{}
	}
}

func RegisterLocalFlockySvc(name oam.FlockyService, endpoint string) {
	flockyServices[name] = endpoint
}

func GetFlockyServiceEndpoint(name oam.FlockyService) string {
	endpoint, found := flockyServices[name]
	if found {
		return endpoint
	} else {
		return ""
	}
}

func GetFlockyServiceEndpoints() map[oam.FlockyService]string {
	return flockyServices
}
