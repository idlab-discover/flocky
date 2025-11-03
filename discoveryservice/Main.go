package main

import (
	"context"
	"fmt"
	"net/http"
	"oamswirly/discoveryservice/config"
	"oamswirly/discoveryservice/discovery"
	discows "oamswirly/discoveryservice/discovery/wsserver"
	"oamswirly/discoveryservice/svcrepository"
	repoclient "oamswirly/discoveryservice/svcrepository/wsclient"
	repows "oamswirly/discoveryservice/svcrepository/wsserver"
	"os"
	"time"

	//_ "net/http/pprof"

	log "github.com/sirupsen/logrus"
)

func main() {
	/*labels := make(map[string]string)
	reqs := v1.ResourceList{}
	reqs[v1.ResourceCPU], _ = resource.ParseQuantity("100m")
	reqs[v1.ResourceMemory], _ = resource.ParseQuantity("100M")
	limits := v1.ResourceList{}
	limits[v1.ResourceCPU], _ = resource.ParseQuantity("500m")
	limits[v1.ResourceMemory], _ = resource.ParseQuantity("300M")
	//labels[oam.MetaConcreteComponentNameProperty] = "Component A"
	cDef := oam.ComponentDef{
		ApiVersion: "v1/beta",
		Kind:       "ComponentDefinition",
		Metadata: oam.Metadata{
			Name:   "Implementation A.1",
			Labels: labels,
		},
		Spec: oam.ComponentDefSpec{
			Workload: oam.WorkloadTypeDescriptor{
				Type: string(oam.ContainerRuntime),
			},
			Schematic: oam.Schematic{
				BaseComponent: "Component A",
				Definition: v1.Container{
					Name:            "TestApp",
					Image:           "gitlab.ilabt.imec.be:4567/fledge/benchmark/dash:1.0.0-capstan",
					ImagePullPolicy: v1.PullAlways,
					Command:         []string{"./dashserver.so"},
					Resources: v1.ResourceRequirements{
						Requests: reqs,
						Limits:   limits,
					},
				},
			},
		},
	}*/
	/*go func() {
		log.Println(http.ListenAndServe("localhost:6060", nil))
	}()*/

	//jsonBytes, _ := json.Marshal(cDef)
	////log.Info(string(jsonBytes))
	log.SetLevel(log.InfoLevel)

	argsWithoutProg := os.Args[1:]
	cfgFile := "defaultConfig.json"
	if len(argsWithoutProg) > 0 {
		cfgFile = argsWithoutProg[0]
	}

	config.LoadConfig(cfgFile)
	ctx := context.Background()
	//sig := make(chan os.Signal, 1)
	//signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	/*go func() {
		<-sig
		cancel()
	}()*/

	go func() {
		discovery.StartDiscovery(ctx, &discovery.SwirlyNodePinger{}, config.Cfg.PingPeriod, config.Cfg.InitialNodes)
	}()

	go func() {
		router := discows.DiscoRouter()
		port := config.Cfg.DiscoAPIPort
		/*if config.Cfg.TestMode {
			nodeNr, _ := strconv.Atoi(config.Cfg.NodeID[1:])
			port += nodeNr
		}*/
		//log.Infof("Hosting disco API %s on port %d", config.Cfg.NodeID, port)
		err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
		if err != nil {
			log.Errorf("Disco HTTP server error %v", err.Error())
		}
	}()
	time.Sleep(100 * time.Millisecond)

	knownNodes, _ := repoclient.GetLocallyKnownSvcNodes(config.Cfg.DiscoAPIPort)
	svcrepository.Start(ctx, knownNodes, config.Cfg.BasicComponentDefs)

	//go func() {
	router := repows.RepoRouter()
	port := config.Cfg.RepoAPIPort
	/*if config.Cfg.TestMode {
		nodeNr, _ := strconv.Atoi(config.Cfg.NodeID[1:])
		port += nodeNr
	}*/
	//log.Infof("Hosting repo API %s on port %d", config.Cfg.NodeID, port)
	err := http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	if err != nil {
		log.Errorf("Repo HTTP server error %v", err.Error())
	}
	//	}()
	//register as node listener on disco API

}
