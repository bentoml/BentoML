template1 = """
from fastapi import FastAPI
from {path} import {class_name}

app=FastAPI()\n
    """
template2 = """
# WARNING:DO NOT EDIT THE BELOW LINE
app.add_api_route(
        path="/{route_path}",
        endpoint={endpoint},
        methods={http_methods},
    )\n
        """


def create_fastapi_file(class_name, module_name, apis_list, store_path):
    path = f"{class_name}.{module_name}"
    complete_template = template1.format(
        path=path,
        class_name=class_name
    )

    for api in apis_list:
        print(api.http_methods,api.name,api.route)
        complete_template += template2.format(
            route_path=api.route,
            endpoint=f"{class_name}.{api.name}",
            http_methods=api.http_methods
        )

    try:
        with open(store_path, "x") as f:
            f.write(complete_template)
    except FileExistsError:
        raise Exception("The FastAPI file already exists")
