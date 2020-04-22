import requests
import os
import json
import yaml
import time
import subprocess
from flask import Flask
from flask import request, jsonify, Response
from daggit.core.base.utils import parse_config

app = Flask(__name__)


@app.route('/daggit/submit', methods=['POST'])
def submit_dag():
    current_wd = os.path.dirname(os.path.realpath(__file__))
    status = 200
    if (request.is_json):
        try:
            req = request.get_json()
            DAGGIT_HOME = req["request"]["input"]["DAGGIT_HOME"]
            APP_HOME = req["request"]["input"]["APP_HOME"]
            job = req["request"]["job"]
            print("******JOB: ", job)
        except:
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
        print("Status:----> ", status)

        if status == 200:
            try:
                # setting environment variables:
                DAGGIT_HOME = os.environ['DAGGIT_HOME']
                APP_HOME = os.environ['APP_HOME']
            except FileNotFoundError:
                raise Exception("Environment variables are not set. Please set the variables!!")
        expt_config = parse_config(path=yaml_loc)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        updated_expt_name = expt_config['experiment_name'] + timestamp
        for k, v in expt_config["inputs"].items():
            if expt_config["inputs"][k].startswith("inputs/"):
                expt_config["inputs"][k] = os.path.join(os.path.split(yaml_loc)[0], v)
        expt_config["experiment_name"] = updated_expt_name
        directory = os.path.join(os.getcwd(), 'data_input', job, timestamp)
        # expt_config["outputs"]["path_to_result_folder"] = directory
        expt_config["inputs"]["path_to_result_folder"] = os.path.join(DAGGIT_HOME, job, timestamp)
        if not os.path.exists(os.path.join(DAGGIT_HOME, job)):
            os.mkdir(os.path.join(DAGGIT_HOME, job))
        if not os.path.exists(os.path.join(DAGGIT_HOME, job, timestamp)):
            os.mkdir(os.path.join(DAGGIT_HOME, job, timestamp))
        print("******DIRECTORY: ", directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        yaml_loc = os.path.join(directory, updated_expt_name + '.yaml')
        with open(yaml_loc, 'w') as f:
            yaml.dump(expt_config, f, default_flow_style=False)
        try:
            # Run daggit
            init_command = "daggit init " + yaml_loc
            subprocess.check_output(init_command, shell=True)
            run_command = "nohup daggit run " + updated_expt_name + " &"
            subprocess.check_output(run_command, shell=True)
            status = Response(status=200)

        except:
            raise ValueError('DAG run fail. Experiment name: ' + updated_expt_name)
            status = Response(status=400)
        time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
        date = time.strftime("%Y-%m-%d")
        api_response = {
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
        return Response(status=400)


@app.route('/daggit/status', methods=['GET'])
def get_dag_status():
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
    api_response = {
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


app.run(host='0.0.0.0', port=3579)

