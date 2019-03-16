import requests
import os
import json
import configparser
import yaml
import time
import subprocess
import pandas as pd
import numpy as np
import pandas as pd
from flask import Flask
from flask import request, jsonify, Response

app = Flask(__name__)

@app.route('/daggit/submit', methods = ['POST'])
def submitDag():
    current_wd = os.path.dirname(os.path.realpath(__file__))

    if (request.is_json):
        try:
            req = request.get_json()
            input = req["params"]["request"]["input"]
            job = req["params"]["request"]["job"]
            content_ids = input["content_ids"]

        except: 
            raise InvalidRequest()
            status = Response(status=400) 
        
        try:
            with open(os.path.join(current_wd, 'expt_name_map.json')) as f:
                name_mapping = json.load(f)
            print("current path", os.path.dirname(os.path.realpath(__file__)))
            yaml_loc = current_wd + name_mapping[job]
            print("yaml location: ", yaml_loc)
        except:
            raise ValueError('Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.')
            status = Response(status=400)
        
        credentials_loc = os.path.join(current_wd, "dag_examples/content_tagging/inputs/credentials.ini")
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(credentials_loc)
        api_key = config["postman credentials"]["api_key"]
        postman_token = config["tagme credentials"]["postman_token"]
        with open(yaml_loc, 'r') as stream:
            expt_config = yaml.load(stream)
        updated_expt_name = expt_config['experiment_name'] + time.strftime("%Y%m%d-%H%M%S")
        expt_config["experiment_name"] = updated_expt_name
        expt_config["inputs"]["categoryLookup"] = os.path.join(current_wd, "dag_examples/content_tagging/inputs/category_lookup.yaml")
        expt_config["inputs"]["pathTocredentials"] = os.path.join(current_wd, "dag_examples/content_tagging/inputs/credentials.ini")

        try:
            directory=os.path.join(os.getcwd(), 'data_input', updated_expt_name)
            if not os.path.exists(directory):
                 os.makedirs(directory)
            df=[]
            for id in content_ids:
                print("id:", id)
                url = "https://api.ekstep.in/composite/v3/search"

                payload = "{\r\n    \"request\": {\r\n        \"filters\":{\r\n            \"identifier\":[\""+id+"\"]\r\n         },\r\n         \"fields\": [\"subject\", \"downloadUrl\", \"language\", \"mimeType\",\"objectType\", \"gradeLevel\", \"artifactUrl\", \"contentType\", \"graph_id\", \"nodeType\", \"node_id\", \"name\", \"description\"],\r\n         \"limit\":1\r\n    }\r\n}"
                headers = {
                    'content-type': "application/json",
                    'authorization': api_key,
                    'cache-control': "no-cache",
                    'postman-token': postman_token
                    }

                response = requests.request("POST", url, data=payload, headers=headers).json()
                try:
                    df.append(response["result"]["content"][0])
                except:
                    pass
            print("df:", df)
            content_df = pd.DataFrame(df)
            content_meta_loc = os.path.join(directory,'content_meta.csv')
            pd.DataFrame(content_df).to_csv(content_meta_loc)

            expt_config["inputs"]["localpathTocontentMeta"] = content_meta_loc
            yaml_loc =os.path.join(directory, updated_expt_name+'.yaml')
            with open(yaml_loc, 'w') as f:
                yaml.dump(expt_config, f, default_flow_style=False)
        except:
            raise ValueError('Unable to write to '+ directory)
            Response(status=401) 
        
        try:
            # Run  daggit
            init_command = "daggit init " + yaml_loc
            subprocess.check_output(init_command, shell=True)
            run_command = "nohup daggit run " + updated_expt_name + " &"
            subprocess.check_output(run_command, shell=True)
            status = Response(status=200) 

        except:
            raise ValueError('DAG run fail. Experiment name: '+ updated_expt_name)
            status = Response(status=400) 
        time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
        date = time.strftime("%Y-%m-%d")
        api_response= {
                "id": "api.org.search",
                "ver": "v1",
                "ts": time_format,
                "params": {
                    "resmsgid": "null",
                    "msgid": "",
                    "err": "null",
                    "status": "success",
                    "errmsg": "null"
                },
                "responseCode": "OK",
                "result": {
                    "status": status,
                    "experiment_name": updated_expt_name,
                    "estimate_time": "",
                    "execution_date": date
                }
            }
        return Response(api_response, status=200)

    else:
        raise InvalidRequest()
        return Response(status=400) 

@app.route('/daggit/status', methods = ['GET'])
def getDagStatus():

    try: 
        expt_name = request.args.get('experiment_name')
    except:
        return Response(status=400) 
    print("*****expt_name:", expt_name)
    try:
        from_time = request.args.get('execution_date')
        command = "airflow dag_state " + expt_name + " " + from_time
        status = subprocess.check_output(command, shell=True)
    except:
        print("Execution date invalid")
    time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
    api_response= {
                "id": "api.daggit.status",
                "ver": "v1",
                "ts": time_format,
                "params": {
                    "resmsgid": "null",
                    "msgid": "",
                    "err": "null",
                    "status": "success",
                    "errmsg": "null"
                },
                "responseCode": "OK",
                "result": {
                    "status": status,
                }
            }
    return Response(api_response, status=200)

 
app.run(host='0.0.0.0', port= 3579)

