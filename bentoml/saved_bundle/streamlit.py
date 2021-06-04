import os

from bentoml.exceptions import BentoMLException
from bentoml.saved_bundle.templates import ENTHIRE_STREAMLIT_TEMPLATE


def create_streamlit_main(fastapi_file: str, fastapi_path: str, store_path: str):
    import sys, importlib
    sys.path.insert(0, fastapi_path)
    mod = importlib.import_module(fastapi_file.split(".")[0])
    try:
        openapi = mod.app.openapi()
    except Exception as e:
        raise BentoMLException(f"Could not get the fastAPI openAPI schema : {e}")
    list_params = []
    for val in openapi['paths'].keys():
        dict_params = {
            "key": val,
            "http_method": list(openapi['paths'][val].keys())[0]
        }
        opi = openapi['paths'][val][dict_params['http_method']]
        if "parameters" in opi:
            ld = []
            for v in opi['parameters']:
                if v['name'] == "self":
                    continue
                ll = {
                    'required': v['required'],
                    'title': v['name']
                }
                if "type" in v['schema']:
                    ll['type'] = v['schema']['type']
                ld.append(ll)
            dict_params['parameters'] = ld
        list_params.append(dict_params)

    key_list = list(openapi['paths'].keys())
    st_upper = "api_selection=st.sidebar.selectbox('Select API',options={})\n\n".format(key_list)
    st_complete = ENTHIRE_STREAMLIT_TEMPLATE + st_upper
    method_head = """\
if api_selection=={what}:
    http_method={http_method}
    parameters={params}
    url={url}    
    """
    method_tail = """\
ll={}
    for param in parameters:
        input_selection = selection(param['title'])
        input = return_input_type(input_selection,param['title'])
        ll[param['title']] = input
    output = requests.request(method=http_method, data=ll, url=url, headers={'Content-Type': 'application/json'})
    st.write(output.content) 
    """
    for val in list_params:
        parms = val['parameters'] if 'parameters' in val else []
        wht = "'" + val['key'] + "'"
        htp_met = "'" + val['http_method'].upper() + "'"
        url = "'" + f"http://backend:8080{val['key']}" + "'"
        ss = method_head.format(what=wht, http_method=htp_met, params=parms, url=url) + method_tail
        st_complete += ss + "\n"

    try:
        with open(os.path.join(store_path, "main.py"), "x") as f:
            f.write(st_complete)
    except Exception as e:
        raise BentoMLException(e)
