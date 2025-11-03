package orchestration

import (
	"oamswirly/common/oam"
	oclients "oamswirly/deploymentservice/orchestration/clients"
	"oamswirly/deploymentservice/wsclient"
	"strconv"

	v1 "k8s.io/api/core/v1"
)

type ServiceClient struct {
	Name          string
	IP            string
	ComponentType oam.Component
	//ComponentImpl oam.ComponentDef
	Migrateable bool
}

type ServiceClientRequest struct {
	Name   string
	IP     string
	SubApp oam.SubApplication
	//ComponentType oam.Component
	//ComponentImpl oam.ComponentDef
}

type ServiceSpec struct {
	ComponentDef oam.ComponentDef
	Deleting     bool
	MinClients   int
	MaxClients   int
}

// indexed by componentdef name! from the orchestration viewpoint, we group by clients that use the same deployment, not the same logical functionality (i.e. component+traits)
var clients map[string][]*ServiceClient
var services map[string]*ServiceSpec
var applications map[string]oam.SubApplication

func Init() {
	clients = make(map[string][]*ServiceClient)
	services = make(map[string]*ServiceSpec)
}

/*func HasService(service string) bool {
	spec, exists := services[service]
	return exists && !spec.Deleting
}*/

func CheckServiceAvailableForClient(client ServiceClientRequest) bool {
	appSpec := client.SubApp.Spec
	cDef := appSpec.ConcreteComponents[0]
	//component := appSpec.Components[0]
	//log.Infof("CheckServiceAvailableForClient %s", cDef.Metadata.Name)
	clients, deployed := clients[cDef.Metadata.Name]
	//Technically, this should also check for traits, but come on man.. that would take another bunch of service calls to the metadata repo, if this node even has one
	if !deployed {
		//log.Info("CheckServiceAvailableForClient not yet deployed, checking resources")
		//load pod yaml for this service
		//pod := parsePodSpec(component, cDef)
		//check resource requirements
		resources := getRequiredResources(cDef.Spec.Schematic.Definition)

		return resourcesFree(resources)
	} else {
		//log.Info("CheckServiceAvailableForClient not yet deployed, checking client limit")
		serviceSpec, _ := services[cDef.Metadata.Name]
		numClients := len(clients)

		if serviceSpec.MaxClients > 0 {
			//log.Infof("CheckServiceAvailableForClient not yet deployed, client limit %v", numClients < serviceSpec.MaxClients)
			return numClients < serviceSpec.MaxClients
		} else {
			//log.Info("CheckServiceAvailableForClient no max client parameter, check OK")
			return true
		}
	}
}

func AddClient(client ServiceClientRequest) bool {
	cDefName := client.SubApp.Spec.ConcreteComponents[0].Metadata.Name
	_, deployed := clients[cDefName]

	if !deployed {
		success := loadRequiredService(client)
		if !success {
			return false
		}
	}
	return addServiceClient(client)
}

func RemoveClient(client ServiceClientRequest) {
	//delete(clients, client.ServiceName)
	cDefName := client.SubApp.Spec.ConcreteComponents[0].Metadata.Name
	sClients, exists := clients[cDefName]
	if !exists {
		return
	}
	newClients := []*ServiceClient{}
	for _, sClient := range sClients {
		if sClient.IP != client.IP {
			newClients = append(newClients, sClient)
		}
	}
	clients[cDefName] = newClients
	//do lower bound clients/resource check
	//not doable on resources obviously, unless a single process is monitored
	//maybe include lower bound on #clients?
	if len(clients[cDefName]) < services[cDefName].MinClients {
		services[cDefName].Deleting = true

		for _, nclient := range clients[cDefName] {
			nclient.Migrateable = false
			wsclient.NotifyInitTeardown(nclient.IP, nclient.Name, client.SubApp.Spec.Components[0])
		}
	}
}

// braindump: the addclient and removeclient mechanism can also be used for an edgenode to assure that it can join/has joined another service provider before
// calling this method, and removeclient easily reverses addclient with no adverse effects (except maybe a migration cascade?)
// find something to avoid that cascade, maybe a flag for removeclient to indicate it was a temporary thing
func MigrationConfirmed(client ServiceClientRequest) {
	cDefName := client.SubApp.Spec.ConcreteComponents[0].Metadata.Name
	if services[cDefName].Deleting {
		//update list of migrated clients
		clientList := clients[cDefName]
		for _, eClient := range clientList {
			if eClient.IP == client.IP {
				eClient.Migrateable = true
			}
		}
		//client.Migrateable = true
		//if all {
		allMigrated := false
		for _, nclient := range clients[cDefName] {
			if !nclient.Migrateable {
				allMigrated = false
			}
		}
		//if ok, migrate
		if allMigrated {
			for _, nclient := range clients[cDefName] {
				wsclient.NotifyTeardown(nclient.IP, nclient.Name, client.SubApp.Spec.Components[0])
			}
			delete(clients, cDefName)
			delete(services, cDefName)
			//pod := parsePodSpec(client.SubApp.Spec.Components[0], client.SubApp.Spec.ConcreteComponents[0])
			oclients.Orch.RemoveSubApp(client.SubApp)
		}
	}
}

func MigrationDenied(client ServiceClientRequest) {
	cDefName := client.SubApp.Spec.ConcreteComponents[0].Metadata.Name
	services[cDefName].Deleting = false

	for _, nclient := range clients[cDefName] {
		nclient.Migrateable = false
	}

	for _, nclient := range clients[cDefName] {
		wsclient.CancelTeardown(nclient.IP, nclient.Name, client.SubApp.Spec.Components[0])
	}
}

func loadRequiredService(client ServiceClientRequest) bool {
	cDefName := client.SubApp.Spec.ConcreteComponents[0].Metadata.Name
	container := client.SubApp.Spec.ConcreteComponents[0].Spec.Schematic.Definition
	//log.Infof("LoadRequiredService %s", cDefName)
	//load pod yaml for this service
	//pod := parsePodSpec(client.SubApp.Spec.Components[0], client.SubApp.Spec.ConcreteComponents[0])
	//check resource requirements
	resources := getRequiredResources(container)

	if resourcesFree(resources) {
		//start service
		//log.Infof("Service %s resources ok, deploying pod", cDefName)
		success := oclients.Orch.DeploySubApp(client.SubApp)
		//have to fix this to generate errors at some point..
		if success {
			//Apply traits
			for _, trait := range client.SubApp.Spec.Components[0].Traits {
				traitOk := tryApplyTrait(trait, client.SubApp.Spec.Components[0])
				if !traitOk {
					success = false
				}
			}
		}

		//log.Infof("Service %s deployed %t", cDefName, success)
		val, found := client.SubApp.Spec.ConcreteComponents[0].Metadata.Labels[oam.MetaMinClients]
		min := 0
		if found {
			min, _ = strconv.Atoi(val)
		}
		val, found = client.SubApp.Spec.ConcreteComponents[0].Metadata.Labels[oam.MetaMaxClients]
		max := 0
		if found {
			max, _ = strconv.Atoi(val)
		}
		//min, _ := strconv.Atoi(pod.ObjectMeta.Labels["minClients"])
		//max, _ := strconv.Atoi(pod.ObjectMeta.Labels["maxClients"])
		spec := ServiceSpec{
			ComponentDef: client.SubApp.Spec.ConcreteComponents[0],
			Deleting:     false,
			MinClients:   min,
			MaxClients:   max,
		}
		services[spec.ComponentDef.Metadata.Name] = &spec
		//register client

		if success {
			clients[cDefName] = []*ServiceClient{} //client}
		}
		return success
		//return success
	} else {
		//server full, deny
		//log.Infof("Service %s resource check failed", cDefName)
		return false
	}
}

func tryApplyTrait(trait oam.Trait, component oam.Component) bool {
	thandler, err := wsclient.GetTraitHandler(trait)
	if err != nil {
		return false
	}
	//There's no handler for this, meaning it's an implicitly applied trait or something used by Flocky itself
	if thandler.LocalEndpoint != "" {
		return true
	}
	result, err := wsclient.TryHandleTrait(thandler.LocalEndpoint, trait)
	if err != nil || result == false {
		return false
	}

	return true
}

func addServiceClient(clientreq ServiceClientRequest) bool {
	cDefName := clientreq.SubApp.Spec.ConcreteComponents[0].Metadata.Name

	if len(clients[cDefName]) < services[cDefName].MaxClients || services[cDefName].MaxClients == 0 {
		//just add, nothing more required
		client := ServiceClient{
			Name:          clientreq.Name,
			IP:            clientreq.IP,
			ComponentType: clientreq.SubApp.Spec.Components[0],
		}
		clients[cDefName] = append(clients[cDefName], &client)
		return true
	} else {
		//server full, deny
		return false
	}
}

/*func getEdgePort(node string) int {
	port := config.Cfg.SwirlyAPIPort
	if config.Cfg.TestMode {
		nodenumber, _ := strconv.Atoi(node[1:])
		port += nodenumber
	}
	return port
}*/

func getRequiredResources(dc v1.Container) map[v1.ResourceName]int {
	resources := make(map[v1.ResourceName]int)
	resources[v1.ResourceCPU] = 0
	resources[v1.ResourceMemory] = 0
	//for _, dc := range pod.Spec.Containers {
	if dc.Resources.Limits == nil {
		dc.Resources.Limits = v1.ResourceList{}
	}
	if dc.Resources.Requests == nil {
		dc.Resources.Requests = v1.ResourceList{}
	}
	memory := dc.Resources.Limits.Memory()
	if memory.IsZero() {
		memory = dc.Resources.Requests.Memory()
	}
	if !memory.IsZero() {
		resources[v1.ResourceMemory] += int(memory.Value())
	}

	cpu := dc.Resources.Limits.Cpu()
	if cpu.IsZero() {
		cpu = dc.Resources.Requests.Cpu()
	}
	if !cpu.IsZero() {
		resources[v1.ResourceCPU] += int(cpu.Value()) * 100
	}
	//}
	return resources
}
