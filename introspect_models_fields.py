import requests
import json

GRAPHQL_ENDPOINT = "https://cms.iamparis.eu/graphql"

# Introspect the GraphQL schema for the 'models' type
introspection_query = '''
query {
  __type(name: "models") {
    fields { name }
  }
}
'''

resp = requests.post(GRAPHQL_ENDPOINT, json={"query": introspection_query})
resp.raise_for_status()
type_info = resp.json()["data"].get("__type", {})
fields = [f["name"] for f in type_info.get("fields", [])]

print("Fields on models:")
print(json.dumps(fields, indent=2))
