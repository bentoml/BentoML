import json

from bentoml.server.api_server import BentoAPIServer

def test_open_api_spec_json_swagger_path(bento_service):
    
    endpoints = ['/feedback', '/healthz', '/metadata', 
        '/metrics', '/predict', '/predict_dataframe', 
        '/predict_dataframe_v1', '/predict_image', '/predict_json', 
        '/predict_multi_images']

    rest_server = BentoAPIServer(bento_service, enable_swagger=False,swagger_url_prefix="")
    test_client = rest_server.app.test_client()
    response = test_client.get("/docs.json")
    path_list = list(response.json['paths'].keys())
    assert all([x in path_list for x in endpoints])

    rest_server = BentoAPIServer(bento_service, enable_swagger=False,swagger_url_prefix="/my_test")
    test_client = rest_server.app.test_client()
    response = test_client.get("/docs.json")
    path_list = list(response.json['paths'].keys())
    assert all(["/my_test" + x in path_list for x in endpoints])

    rest_server = BentoAPIServer(bento_service, enable_swagger=False,swagger_url_prefix="my_test")
    test_client = rest_server.app.test_client()
    response = test_client.get("/docs.json")
    path_list = list(response.json['paths'].keys())
    assert all(["/my_test" + x in path_list for x in endpoints])

    rest_server = BentoAPIServer(bento_service, enable_swagger=False,swagger_url_prefix="////my_test//")
    test_client = rest_server.app.test_client()
    response = test_client.get("/docs.json")
    path_list = list(response.json['paths'].keys())
    assert all(["/my_test" + x in path_list for x in endpoints])