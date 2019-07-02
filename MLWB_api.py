import requests
import os
import pip
import json
import configparser
import yaml
import time
import subprocess
import pandas as pd
import numpy as np
from flask import Flask
from flask import request, jsonify, Response

app = Flask(__name__)

@app.route('/daggit/submit', methods = ['POST'])
def submitDag():
    errmsg= ""
    status = ""
    yaml_loc = ""
    current_wd = os.path.dirname(os.path.realpath(__file__))
    print(current_wd )
    if (request.is_json):
        try:
            req = request.get_json()
            input = req["request"]["input"]
            job = req["request"]["experiment_name"]

        except:
            #raise Exception("InvalidRequest")
            status = 400
            errmsg = "InvalidRequest"

        try:
            with open(os.path.join(current_wd, 'expt_name_map.json')) as f:
                name_mapping = json.load(f)
            print("current path", os.path.dirname(os.path.realpath(__file__)))
            yaml_loc = current_wd + name_mapping[job]
            print("yaml location: ", yaml_loc)
        except:
            #raise ValueError('Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.')
            status = 400
            errmsg = 'Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.'

        if yaml_loc:
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
            expt_config["inputs"]["DS_DATA_HOME"] = os.environ['DS_DATA_HOME']
            try:
                directory=os.path.join(os.getcwd(), 'data_input', updated_expt_name)
                if not os.path.exists(directory):
                     os.makedirs(directory)
                df=[]


                content_meta_loc = os.path.join(directory,'content_meta.csv')
                pd.DataFrame(content_df).to_csv(content_meta_loc)

                expt_config["inputs"]["localpathTocontentMeta"] = content_meta_loc
                yaml_loc =os.path.join(directory, updated_expt_name+'.yaml')
                with open(yaml_loc, 'w') as f:
                    yaml.dump(expt_config, f, default_flow_style=False)
                status = 200
            except:
                raise ValueError('Unable to write to '+ directory)
                status = 401
                errmsg= 'Unable to write to '+ directory
        else:
            updated_expt_name = ""
            estimate_time = ""

        time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
        date = time.strftime("%Y-%m-%d")

        if status ==200:
            param_status="success"
        else:
            param_status="fail"
        api_response= {
                "id": "api.org.search",
                "ver": "v1",
                "ts": time_format,
                "params": {
                    "resmsgid": "null",
                    "msgid": "",
                    "err": "null",
                    "status": param_status,
                    "errmsg": errmsg
                },
                "responseCode": "OK",
                "result": {
                    "status": status,
                    "experiment_name": updated_expt_name,
                    "estimate_time": "",
                    "execution_date": date
                }
            }
        print(api_response)
        return jsonify(api_response)

        try:
            # Run  daggit
            init_command = "daggit init " + yaml_loc
            subprocess.check_output(init_command, shell=True)
            run_command = "nohup daggit run " + updated_expt_name + " &"
            subprocess.check_output(run_command, shell=True)


        except:
            raise ValueError('DAG run fail. Experiment name: '+ updated_expt_name)
            return Response(status=400)

    else:
        raise InvalidRequest()
        return Response(status=400)

@app.route('/daggit/status', methods = ['GET'])
def getDagStatus():

    expt_name = request.args.get('experiment_name')
    errmsg=""
    if expt_name:
        print("expt_name:", expt_name)

        from_time = request.args.get('execution_date')
        if not from_time:
            print("Execution date invalid. Defaulting to 'today'")
            from_time = time.strftime("%Y-%m-%d")

        command = "airflow dag_state " + expt_name + " " + from_time

        try:
            status = subprocess.check_output(command, shell=True)
        except:
            print(command)
            status = 400
            errmsg = command+ " failed"
    else:
        status = 400
        errmsg = "experiment_name required"

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
                    "errmsg": errmsg
                },
                "responseCode": "OK",
                "result": {
                    "status": status,
                }
            }
    return jsonify(api_response)


app.run(host='0.0.0.0', port= 3579)

