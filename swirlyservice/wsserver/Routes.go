package wsserver

import (
	"github.com/gorilla/mux"

	"net/http"
)

type Route struct {
	Name        string
	Method      string
	Pattern     string
	HandlerFunc http.HandlerFunc
	Queries     []string
}

type Routes []Route

func SwirlyRouter() *mux.Router {

	router := mux.NewRouter().StrictSlash(true)
	for _, route := range routes {
		router.
			Methods(route.Method).
			Path(route.Pattern).
			Name(route.Name).
			Handler(route.HandlerFunc)
		//Queries(route.Queries)
	}

	return router
}

var routes = Routes{
	// Repo callback API
	Route{
		Name:        "nodeAppsChanged",
		Method:      "POST",
		Pattern:     "/nodeAppsChanged",
		HandlerFunc: NodeAppsChanged,
		Queries:     []string{},
	},
	Route{
		Name:        "nodeCapsChanged",
		Method:      "POST",
		Pattern:     "/nodeCapsChanged",
		HandlerFunc: NodeCapsChanged,
		Queries:     []string{},
	},
	// Swirly orchestration API
	Route{
		Name:        "deployApplication",
		Method:      "POST",
		Pattern:     "/deployApplication",
		HandlerFunc: DeployApplication,
		Queries:     []string{},
	},
	Route{
		Name:        "deleteApplication",
		Method:      "POST",
		Pattern:     "/deleteApplication",
		HandlerFunc: DeleteApplication,
		Queries:     []string{},
	},
	Route{
		Name:        "tryMigrate",
		Method:      "POST",
		Pattern:     "/tryMigrate",
		HandlerFunc: TryMigrate,
		Queries:     []string{},
	},
	Route{
		Name:        "cancelMigrate",
		Method:      "POST",
		Pattern:     "/cancelMigrate",
		HandlerFunc: CancelMigrate,
		Queries:     []string{},
	},
	Route{
		Name:        "migrate",
		Method:      "POST",
		Pattern:     "/migrate",
		HandlerFunc: Migrate,
		Queries:     []string{},
	},
}
