package oam

import (
	"net"

	v1 "k8s.io/api/core/v1"
)

type ScopeDef struct {
	ApiVersion string
	Kind       string
	Metadata
	Spec ScopeSpec
}

type ScopeSpec struct {
	AllowComponentOverlap bool
	DefinitionRef
}

//NetworkScope not supported in Feather (everything's combined as a pod anyway), so no point in defining it

type StorageScope struct {
	ApiVersion string
	Kind       string
	Metadata
	Spec StorageScopeSpec
}

type SidecarScope struct {
	ApiVersion string
	Kind       string
	Metadata
	Spec SidecarScopeSpec
}

type StorageScopeSpec struct {
	Volumes []v1.Volume
}

type NetworkScope struct {
	ApiVersion string
	Kind       string
	Metadata
	Spec NetworkScopeSpec
}

type NetworkScopeSpec struct {
	NodePodCIDR   net.IP
	ApplicationIP net.IP
}

type SidecarScopeSpec struct {
	Id string
}
