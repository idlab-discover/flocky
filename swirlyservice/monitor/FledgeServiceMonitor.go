package monitor

import (
	"oamswirly/swirlyservice/wsclient"
	"time"
)

type FledgeServiceMonitor struct {
	deployCallback  func(name string)
	removeCallback  func(name string)
	running         bool
	runningServices map[string]bool
}

func (sm *FledgeServiceMonitor) Init(svcToMonitor []string) ServiceMonitor {
	sm.running = true

	sm.runningServices = make(map[string]bool)

	for _, service := range svcToMonitor {
		sm.runningServices[service] = false
	}

	go func() { sm.monitorFledge() }()
	return sm
}

func (sm *FledgeServiceMonitor) Stop() {
	sm.running = false
}

func (sm *FledgeServiceMonitor) monitorFledge() {
	time.Sleep(time.Second * 60)
	for sm.running {
		pods, err := wsclient.GetDeployedPods()

		if err != nil {

		}

		//check for new pods
		for _, pod := range pods {
			running, shouldMonitor := sm.runningServices[pod.Name]
			if shouldMonitor && !running {
				sm.runningServices[pod.Name] = true
				sm.deployCallback(pod.Name)
			}
		}

		//check if any stopped
		for service, running := range sm.runningServices {
			if running {
				inPods := false
				for _, pod := range pods {
					if pod.Name == service {
						inPods = true
					}
				}

				if !inPods {
					sm.runningServices[service] = false
					sm.removeCallback(service)
				}
			}
		}

		time.Sleep(5 * time.Second)
	}
}

func (sm *FledgeServiceMonitor) ServiceDeployedCallback(cb func(name string)) {
	sm.deployCallback = cb
}

func (sm *FledgeServiceMonitor) ServiceRemovedCallback(cb func(name string)) {
	sm.removeCallback = cb
}
