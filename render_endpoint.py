import os
import requests
import json
from flask import Flask
from flask import request, jsonify, Response

app = Flask(__name__)

current_wd = os.path.dirname(os.path.realpath(__file__))
apps = os.listdir(os.path.join(current_wd,"dag_examples"))

endpoint_dict={}
for item in apps:
    config_path = os.path.join(current_wd,"dag_examples",item,"config.json")
    try:
        with open(config_path) as f:
            endpoint_mapping = json.load(f)
        for endpoint in name_mapping.keys():

            @app.route(endpoint, methods = ['POST'])

            module_name = ".".join(["dag_examples",item,"microservice"])
            module = __import__(module_name)
            method_to_call = getattr(module, 'endpoint_mapping[endpoint]')
            if (request.is_json):
                try:
                    req = request.get_json()
                    result = method_to_call(req)
                except:
                    raise ValueError('Invalid method call for '+endpoint_mapping[endpoint])
                    return Response(status=400)
            else:
                raise InvalidRequest()
                return Response(status=400)
    except:
        pass
