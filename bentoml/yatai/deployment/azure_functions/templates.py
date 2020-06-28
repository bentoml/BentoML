AZURE_API_FUNCTION_JSON = """\
{{
  "scriptFile": "__init__.py",
  "bindings": [
    {{
      "authLevel": "{function_auth_level}",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": [
        "get",
        "post",
        "put",
        "delete",
        "patch"
      ],
      "route": "{{*route}}"
    }},
    {{
      "type": "http",
      "direction": "out",
      "name": "$return"
    }}
  ]
}}
"""
