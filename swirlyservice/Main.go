package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"oamswirly/common/oam"
	"oamswirly/swirlyservice/config"
	"oamswirly/swirlyservice/orchestration"
	"oamswirly/swirlyservice/wsclient"
	"oamswirly/swirlyservice/wsserver"
	"os"
	"strconv"

	log "github.com/sirupsen/logrus"
)

func main() {
	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultconfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)

	app := oam.Application{
		ApiVersion: "v1/beta",
		Kind:       "Application",
		Metadata: oam.Metadata{
			Name: "TestApp",
		},
		Spec: oam.ApplicationSpec{
			Components: []oam.Component{
				oam.Component{
					Name:       "First component",
					Type:       "Component A",
					Properties: oam.Properties{},
					Traits: []oam.Trait{
						oam.Trait{
							Type: string(oam.GreenEnergyTraitType),
							Properties: oam.Properties{
								oam.RequiredTraitProperty: "true",
							},
						},
						oam.Trait{
							Type: string(oam.NodeNetworkEncryptionTraitType),
							Properties: oam.Properties{
								oam.RequiredTraitProperty: "true",
							},
						},
					},
					Scopes: make(map[string]string),
				},
				oam.Component{
					Name:       "Second component",
					Type:       "Component B",
					Properties: oam.Properties{},
					Traits: []oam.Trait{
						oam.Trait{
							Type: string(oam.SecureRuntimeTraitType),
							Properties: oam.Properties{
								oam.RequiredTraitProperty: "true",
							},
						},
					},
					Scopes: make(map[string]string),
				},
			},
		},
	}
	jsonBytes, _ := json.Marshal(app)
	fmt.Println(string(jsonBytes))
	/*sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		//rootContextCancel()
		common.Stop()
	}()*/

	orchestration.InitMesher()

	/*go func() {
		cdiscovery.StartDiscovery(&discovery.EdgeNodePinger{}, config.Cfg.PingPeriod, config.Cfg.InitialNodes)
	}()*/
	//register as node update listener
	wsclient.RegisterAsNodeUpdateListener(config.Cfg.NodeID)

	/*go func() {
		svcMap := config.Cfg.SupportComponents
		toMonitor := []string{}
		for svc, _ := range svcMap {
			toMonitor = append(toMonitor, svc)
		}

		orchType := config.Cfg.ServiceMonitorType
		switch orchType {
		case "fledge":
			monitor.Monitor = &monitor.FledgeServiceMonitor{}
		default:
			monitor.Monitor = &monitor.FledgeServiceMonitor{}
		}

		monitor.Monitor.ServiceDeployedCallback(orchestration.DeploySupportComponentsFor)
		monitor.Monitor.ServiceRemovedCallback(orchestration.RemoveSupportComponentsFor)
		monitor.Monitor.Init(toMonitor)
	}()*/

	orchType := config.Cfg.ServiceLocatorType
	switch orchType {
	case "hosts":
		orchestration.Locator = (&orchestration.HostsComponentLocator{}).Init()
	default:
		orchestration.Locator = (&orchestration.HostsComponentLocator{}).Init()
	}

	router := wsserver.SwirlyRouter()
	port := config.Cfg.SwirlyAPIPort
	if config.Cfg.TestMode {
		nodeNr, _ := strconv.Atoi(config.Cfg.NodeID[1:])
		port += nodeNr
	}

	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf("HTTP server error %v", err)
	}
}
