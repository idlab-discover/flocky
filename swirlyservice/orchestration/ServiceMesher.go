package orchestration

import (
	"errors"
	"fmt"
	"oamswirly/common/oam"
	"oamswirly/swirlyservice/config"
	"oamswirly/swirlyservice/orchestration/qoe"
	"oamswirly/swirlyservice/wsclient"
	"time"

	log "github.com/sirupsen/logrus"
)

type ComponentProvider struct {
	qoe.ActiveProvider
	ApplicationIds []string
	//oam.Component
	//oam.ComponentDef
}

// By component name, because if two apps need the same component we don't need to redeploy it
var componentProviders map[string][]ComponentProvider
var tempMigrateNodes map[string]ComponentProvider

// By node name
var knownNodes map[string]oam.NodeSummary

func InitMesher() {
	componentProviders = make(map[string][]ComponentProvider)
	tempMigrateNodes = make(map[string]ComponentProvider)
	knownNodes = make(map[string]oam.NodeSummary)
	qoe.Evaluator = qoe.LegacyEvaluator{}
	//discovery.NodesUpdatedCallback = ProcessUpdatedPings
}

func NodeResourcesUpdated(capsSummary []oam.NodeSummary) {
	//log.Info("Node resource update received, processing")
	doUpdate := false
	for _, capSummary := range capsSummary {
		//log.Infof("Processing node %s", capSummary.Name)
		nodeSummary, exists := knownNodes[capSummary.Name]
		if exists {
			//log.Infof("Existing node %s, updating", capSummary.Name)
			//check for meaningful differences first
			//resources, if >5% diff then we need to reconfigure component deployments

			//traits, if any changed then we need to reconfigure component deployments
			if len(oam.TraitDefArrayDiff(nodeSummary.NodeCaps.Caps.SupportedTraits, capSummary.NodeCaps.Caps.SupportedTraits)) > 0 {
				doUpdate = true
			}
			//workloads, if any changed then we need to reconfigure component deployments
			if len(oam.WorkloadArrayDiff(nodeSummary.NodeCaps.Caps.SupportedWorkloads, capSummary.NodeCaps.Caps.SupportedWorkloads)) > 0 {
				doUpdate = true
			}

			nodeSummary.NodeCaps = capSummary.NodeCaps
			knownNodes[capSummary.Name] = nodeSummary
		} else {
			//log.Infof("Nonexisting node %s, assigning", capSummary.Name)
			knownNodes[capSummary.Name] = capSummary
		}
	}
	//but we don't do it immediately, because node app updates might be incoming around this period so wait around 1s to buffer stuff
	if doUpdate {
		go func() {
			time.Sleep(time.Millisecond * 50)
			ProcessUpdatedNodeCaps()
		}()
	}
}

func NodeAppsUpdated(appsSummary []oam.NodeSummary) {
	//log.Info("Node apps updated, processing")
	for _, appSummary := range appsSummary {
		//log.Infof("Processing node %s", appSummary.Name)
		nodeSummary, exists := knownNodes[appSummary.Name]
		if exists {
			//log.Infof("Existing node %s, updating", appSummary.Name)
			nodeSummary.NodeApps = appSummary.NodeApps
			knownNodes[appSummary.Name] = nodeSummary
		} else {
			//log.Infof("Nonexisting node %s, assigning", appSummary.Name)
			knownNodes[appSummary.Name] = nodeSummary
		}
	}
}

// func GetFogServices() map[string]oam.NodeSummary {
// 	return fogServices
// }

// There's a few cases here:
// Either the subapp is new and the remote node didn't run it yet, so we add an entirely new componentprovider and set existinginstance to the new subapp
// Either the subapp already existed on the remote node, but it's new to us, so we still add a new componentprovider but existinginstance will already be set (we got that from the metadata store)
// Either we were already running the subapp and the remote node had already started it, so we just add appid and keep all else
// If somehow the subapp were already "running" but the remote node didn't get that memo so existininstance isn't set, this thing will trigger the end of the world.
func checkAddComponentProvider(subapp oam.SubApplication, node *qoe.ActiveProvider, appId string) {
	component := subapp.Spec.Components[0]
	cDef := subapp.Spec.ConcreteComponents[0]
	//log.Infof("Adding component %s concrete def %s to providers for appId %s", component.Name, cDef.Metadata.Name, appId)
	componentName := cDef.Spec.Schematic.BaseComponent

	if node.ExistingInstance == nil {
		node.ExistingInstance = &subapp
	}
	cProvider := ComponentProvider{
		ActiveProvider: *node,
		//Component:      component,
		//ComponentDef:   concreteDef,
		ApplicationIds: []string{appId},
	}

	cProviders, exists := componentProviders[componentName]
	if exists {
		//log.Debugf("Same name component provider already registered")
		if eProvider := findMatchingComponentProvider(cProviders, component.Traits); eProvider != nil {
			log.Debug("Exact component def provider already registered, adding application id")
			eProvider.ApplicationIds = append(eProvider.ApplicationIds, appId)
		} else {
			//log.Debugf("Adding component def to component providers")
			cProviders = append(cProviders, cProvider)
		}
	} else {
		//log.Debugf("Creating new component providers collection for %s", component.Name)
		cProviders = []ComponentProvider{
			cProvider,
		}
	}
	componentProviders[componentName] = cProviders
}

func findMatchingComponentProvider(cProviders []ComponentProvider, traits []oam.Trait) *ComponentProvider {
	//log.Debugf("Finding matching component provider for %d traits", len(traits))
	for _, provider := range cProviders {
		if oam.TraitsEqual(provider.ExistingInstance.Spec.Components[0].Traits, traits) {
			//log.Debugf("Matching component provider found with concrete def %s at node %s", provider.ExistingInstance.Spec.ConcreteComponents[0].Metadata.Name, provider.Node.Name)
			return &provider
		}
	}
	return nil
}

func findMatchingComponentProviderByApplication(cProviders []ComponentProvider, appId string) *ComponentProvider {
	//log.Debugf("Finding matching component provider for application id %s", appId)
	for _, provider := range cProviders {
		for _, app := range provider.ApplicationIds {
			if app == appId {
				//log.Debugf("Matching component provider found with concrete def %s at node %s", provider.ExistingInstance.Spec.ConcreteComponents[0].Metadata.Name, provider.Node.Name)
				return &provider
			}
		}
	}
	return nil
}

func DeploySupportComponentsFor(application oam.Application) error {
	log.Infof("%d Deploy support components for %s", time.Now().UnixMilli(), application.Metadata.Name)
	//supportComponents := config.Cfg.SupportComponents[application]
	var generalErr error = nil
	failed := []*qoe.ActiveProvider{}
	for _, component := range application.Spec.Components {
		success := false
		var bestNode *qoe.ActiveProvider
		var subapp *oam.SubApplication //oam.ComponentDef
		for !success {
			bestNode, subapp = findBestComponentProvider(component, failed)
			if bestNode != nil {
				success = wsclient.TryRegisterClientWithNode(bestNode.Node.NetInfo.PrimaryAddress, bestNode.Node.Name, *subapp)
			} else {
				failed = append(failed, bestNode)
				break
			}
		}
		if bestNode == nil {
			log.Errorf("%d Couldn't register component with suitable node", time.Now().UnixMilli())
			return errors.New("Couldn't register component with suitable node")
		}
		//Here, we use the concrete component name, because we might require other concrete types of the same component too depending on traits
		checkAddComponentProvider(*subapp, bestNode, application.Metadata.Name)
		log.Infof("%d Component provider registered", time.Now().UnixMilli())
		//Using:
		//COMPONENT DEPENDENCY SCOPES!
		//Figure out which components should be deployed in groups due to interdependencies (if any)
		//Then force deploy those together (which makes things a little more annoying in the orchestration algo, but we'll figure it out)
		//Main idea: componentProviders[svc] is no longer the goto node if it exists (unless there's just ONE component, or all return the same node)
		//instead, if the existing component providers for a group are different nodes, those nodes should be attempted to deploy the other service on first
		//(i.e. swap one of the components to the same node as the other)
		//Not implemented yet, maybe I should release a bunch of this first instead of going down even more rabbit holes

		//This of course is where it goes completely to shit, no idea how to handle this yet, because there's only one hosts file no matter what
		//Probably have to feed it into a custom DNS program, which then magicks it up
		err := updateServerFor(component.Type, bestNode.Node.NetInfo.PrimaryAddress)
		log.Infof("%d DNS info for component registered", time.Now().UnixMilli())
		if err != nil {
			log.Errorf("Failed to locally register component provider %v", err)
			generalErr = err
		}
	}
	return generalErr
}

func findBestComponentProvider(component oam.Component, except []*qoe.ActiveProvider) (*qoe.ActiveProvider, *oam.SubApplication) {
	log.Infof("%d Finding component provider for %s", time.Now().UnixMilli(), component.Name)
	// //don't need to find one if it's already prepared for another deployment (unless we need another one, "except")

	cDefs, err := wsclient.GetConcreteComponentsFor(component)
	if err != nil {
		log.Errorf("%d Failed getting concrete components for %s, %v", time.Now().UnixMilli(), component.Name, err)
	}

	var suitableProvider *ComponentProvider
	//for _, cDef := range cDefs {
	existingProviders, known := componentProviders[component.Name] //cDef.Spec.Schematic.BaseComponent]
	if known {
		log.Infof("%d Checking existing component providers for %s", time.Now().UnixMilli(), component.Name)
		/*if suitableProvider = findMatchingComponentProvider(existingProviders, component.Traits); suitableProvider != nil {
			break
		}*/
		suitableProvider = findMatchingComponentProvider(existingProviders, component.Traits)
	} else {
		log.Infof("%d No existing component provider for %s", time.Now().UnixMilli(), component.Name)
	}
	//}
	//First, see if we already have a suitable cdef deployed somewhere for this component
	if suitableProvider != nil {
		log.Infof("%d Returning existing provider %s", time.Now().UnixMilli(), suitableProvider.Node.Name)
		return &suitableProvider.ActiveProvider, suitableProvider.ExistingInstance
	}
	// Otherwiiiise, get cracking

	//Check if the oam repository even contains a ComponentDef for the Component to deploy, with a WorkloadType that suits the required traits (e.g. secureruntime..)
	//concreteComponentDefs, err := wsclient.GetConcreteComponentsFor(component)
	//Now, we're going to make some messy code. It would be better to refactor this loop into a different method, but let's face it, we just want the best one, and it should already
	//be sorted for best ones first, so if we just iterate we *will* get the best outcome.

	nodeSummaries := []oam.NodeSummary{}
	for _, node := range knownNodes {
		nodeSummaries = append(nodeSummaries, node)
	}
	log.Infof("%d Checking %d known nodes", time.Now().UnixMilli(), len(nodeSummaries))

	for _, cDef := range cDefs {
		bestSuitableNodes := qoe.OrderByBestComponentProvider(component, cDef, nodeSummaries, []oam.Component{})
		for _, node := range bestSuitableNodes {
			log.Infof("%d Trying to register service with node %s", time.Now().UnixMilli(), node.Node.Name)
			//
			subapp := oam.SubApplication{
				ApiVersion: "v1/beta",
				Kind:       "SubApplication",
				Metadata: oam.Metadata{
					Name: fmt.Sprintf("%s-%s", config.Cfg.NodeID, cDef.Metadata.Name),
				},
				Spec: oam.SubApplicationSpec{
					Components:         []oam.Component{component},
					ConcreteComponents: []oam.ComponentDef{cDef},
				},
			}
			if node.ExistingInstance != nil {
				subapp = *node.ExistingInstance
			}
			//found := wsclient.TryRegisterClientWithNode(node.Node.PrimaryAddress, node.Node.Name, component, cDef)
			found := wsclient.CheckNodeAvailable(node.Node.NetInfo.PrimaryAddress, node.Node.Name, subapp)
			if found {
				log.Infof("%d %s available for service %s", time.Now().UnixMilli(), node.Node.Name, component.Type)
				return &node, &subapp
			}
		}
	}
	return nil, nil
}

func RemoveSupportComponentsFor(application oam.Application) {
	//supportComponents := config.Cfg.SupportComponents[application]

	//do a check to see if these aren't required by other edge services
	//We don't need to check like this anymore, better to check with "decent" code, and appIDs are tracked by component providers now
	/*for app, supportCmps := range config.Cfg.SupportComponents {
		if app != application {
			supportComponents = exclude(supportComponents, supportCmps)
		}
	}*/

	toDelete := []ComponentProvider{}
	for _, svc := range application.Spec.Components {
		provider := findMatchingComponentProviderByApplication(componentProviders[svc.Type], application.Metadata.Name)
		if provider != nil {
			newAppIds := []string{}
			for _, appId := range provider.ApplicationIds {
				if appId != application.Metadata.Name {
					newAppIds = append(newAppIds, appId)
				}
			}
			provider.ApplicationIds = newAppIds

			if len(newAppIds) == 0 {
				toDelete = append(toDelete, *provider)
			}
		}
	}

	go func() {
		for _, provider := range toDelete {
			removeFromSupportComponent(provider)
		}
	}()
}

/*func exclude(list []oam.Component, exclude []oam.Component) []oam.Component {
	newList := []oam.Component{}

	for _, component := range list {
		include := true
		for _, eComponent := range exclude {
			if component.Type == eComponent.Type && oam.TraitsEqual(component.Traits, eComponent.Traits) {
				include = false
			}
		}
		if include {
			newList = append(newList, component)
		}
	}

	return newList
}*/

func removeFromSupportComponent(provider ComponentProvider) {
	component := provider.ExistingInstance.Spec.Components[0]
	removeProviderFromSupportComponent(provider)
	providers := componentProviders[component.Type]
	providersLeft := []ComponentProvider{}
	for _, oProvider := range providers {
		if !providersEqual(provider, oProvider) {
			providersLeft = append(providersLeft, oProvider)
		}
	}

	if len(providersLeft) == 0 {
		delete(componentProviders, component.Type)
	} else {
		componentProviders[component.Type] = providersLeft
	}
}

func removeProviderFromSupportComponent(provider ComponentProvider) bool {
	success := wsclient.TryRemoveFromNode(provider.Node.NetInfo.PrimaryAddress, provider.Node.Name, *provider.ExistingInstance)

	tries := 0
	for !success && tries < 10 {
		time.Sleep(5 * time.Second)
		success = wsclient.TryRemoveFromNode(provider.Node.NetInfo.PrimaryAddress, provider.Node.Name, *provider.ExistingInstance)
		tries++
	}
	return success
}

func providersEqual(provider, oProvider ComponentProvider) bool {
	return (provider.Node.Name == oProvider.Node.Name) && oam.TraitsEqual(provider.ExistingInstance.Spec.Components[0].Traits, oProvider.ExistingInstance.Spec.Components[0].Traits)
}

// don't forget periodic ping updates that can influence deployed services... implement somehow
func ProcessUpdatedNodeCaps() {
	go func() {
		components := []string{}
		for svc, _ := range componentProviders {
			components = append(components, svc)
		}
		//for svc, providers := range componentProviders {
		for _, component := range components {
			newProviders := []ComponentProvider{}
			providers := componentProviders[component]
			for _, provider := range providers {
				//TODO this needs to be done with evaluators instead
				//dist := getNodeQoEIfKnown(knownNodes, provider.ActiveProvider.Node)
				//only do something about it if we crossed the maxping threshold
				//if dist == -1 || dist > config.Cfg.MaxPing*2 || (dist > config.Cfg.MaxPing && fnode.Latency < config.Cfg.MaxPing) {
				if !qoe.RevalidateNode(provider.ActiveProvider, provider.ExistingInstance.Spec.Components[0], provider.ExistingInstance.Spec.ConcreteComponents[0]) { //dist == -1 || dist > config.Cfg.MaxPing*2 {
					//this means we can probably get a better node, and should try

					newProvider, subapp := findBestComponentProvider(provider.ExistingInstance.Spec.Components[0], nil)
					success := false
					if newProvider != nil { //&& newProvider.Distance < dist {
						removeFromSupportComponent(provider)
						success = wsclient.TryRegisterClientWithNode(newProvider.Node.NetInfo.PrimaryAddress, newProvider.Node.Name, *subapp)
					}
					if success {
						provider.ActiveProvider = *newProvider
						provider.ExistingInstance = subapp
						//checkAddComponentProvider(provider.Component, componentDef, newProvider, provider.ApplicationIds)
						updateServerFor(component, newProvider.Node.NetInfo.PrimaryAddress)
					}
				}
				newProviders = append(newProviders, provider)
			}
			componentProviders[component] = newProviders
		}
	}()
}

func TryMigrateComponent(component oam.Component) {
	success := false
	var bestNode *qoe.ActiveProvider
	var subapp *oam.SubApplication
	for !success {
		bestNode, subapp = findBestComponentProvider(component, nil)
		if bestNode != nil {
			success = wsclient.TryRegisterClientWithNode(bestNode.Node.NetInfo.PrimaryAddress, bestNode.Node.Name, *subapp)
		} else {
			break
		}
	}

	//bestNode := findBestComponentProvider(component, nil)
	providers := componentProviders[component.Type]
	provider := findMatchingComponentProvider(providers, component.Traits)
	if bestNode != nil {
		bestNode.ExistingInstance = subapp
		tempMigrateNodes[component.Type] = ComponentProvider{
			ActiveProvider: *bestNode,
			ApplicationIds: provider.ApplicationIds,

			//Component:      component,
			//ComponentDef:   cDef,
		}
		//notify current fog node to migrate
		wsclient.NotifyMigrateSuccess(provider.Node.NetInfo.PrimaryAddress, provider.Node.Name, *subapp)
	} else {
		//notify current fog node to cancel
		wsclient.NotifyMigrateFailed(provider.Node.NetInfo.PrimaryAddress, provider.Node.Name, *subapp)
	}
}

func MigrateComponent(component oam.Component) {
	//no need to remove from current node, it does that while notifying all its clients
	cProvider := tempMigrateNodes[component.Type]

	newProviders := []ComponentProvider{}
	for _, provider := range componentProviders[component.Type] {
		if oam.TraitsEqual(provider.ExistingInstance.Spec.Components[0].Traits, cProvider.ExistingInstance.Spec.Components[0].Traits) {
			newProviders = append(newProviders, provider)
			/*provider.ActiveProvider = cProvider.ActiveProvider
			provider.ComponentDef = cProvider.ComponentDef
			updateServerFor(component.Name, cProvider.Node.PrimaryAddress)*/
		}
	}
	newProviders = append(newProviders, cProvider)
	updateServerFor(component.Type, cProvider.Node.NetInfo.PrimaryAddress)
	componentProviders[component.Type] = newProviders

	delete(tempMigrateNodes, component.Type)
}

func CancelMigrate(component oam.Component) {
	cProvider := tempMigrateNodes[component.Type]
	wsclient.TryRemoveFromNode(cProvider.Node.NetInfo.PrimaryAddress, cProvider.Node.Name, *cProvider.ExistingInstance)

	delete(tempMigrateNodes, component.Type)
}

func updateServerFor(component string, ip string) error {
	return Locator.UpdateComponentLocation(component, ip)
}
