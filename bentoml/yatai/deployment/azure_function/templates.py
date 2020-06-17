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

BENTO_SERVICE_AZURE_FUNCTION_DOCKERFILE = """\
FROM bentoml/azure-function:{bentoml_version}

COPY . /home/site/wwwroot

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /home/site/wwwroot/bentoml-init.sh ]; then /bin/bash -c /home/site/wwwroot/bentoml-init.sh; fi
"""  # noqa: E501
