package monitor

type ServiceMonitor interface {
	Init(svcToMonitor []string) ServiceMonitor
	Stop()
	ServiceDeployedCallback(func(name string))
	ServiceRemovedCallback(func(name string))
}

var Monitor ServiceMonitor
