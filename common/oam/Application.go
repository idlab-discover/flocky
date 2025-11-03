package oam

type Application struct {
	ApiVersion string
	Kind       string
	Metadata   Metadata
	Spec       ApplicationSpec
}

type ApplicationSpec struct {
	Components []Component
}

type SubApplication struct {
	ApiVersion string
	Kind       string
	Metadata   Metadata
	Spec       SubApplicationSpec
}

type SubApplicationSpec struct {
	Components         []Component
	ConcreteComponents []ComponentDef
	StorageScope       StorageScope
	NetworkScope       NetworkScope
}

var MetaConcreteComponentName = "ConcreteComponentName"

type Component struct {
	Name       string
	Type       string
	Properties Properties
	Traits     []Trait
	Scopes     map[string]string
}

type Trait struct {
	Type       string
	Properties Properties
}

type Properties map[string]interface{}

var CommandComponentProperty = "command"
var ArgsCommandProperty = "args"
