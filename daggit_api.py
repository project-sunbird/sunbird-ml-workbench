import json
import os
import subprocess
import time

import yaml
from daggit.core.base.utils import parse_config
from flask import Flask
from flask import request, Response,jsonify

app = Flask(__name__)


@app.route('/daggit/submit', methods=['POST'])
def submit_dag():
    current_wd = os.path.dirname(os.path.realpath(__file__))
    status = 200
    if (request.is_json):
        try:
            req = request.get_json()
            APP_HOME = req["request"]["input"]["APP_HOME"]
            job = req["request"]["job"]
            print("******JOB: ", job)
        except:
            status = 400

        try:
            with open(os.path.join(current_wd, 'expt_name_map.json')) as f:
                name_mapping = json.load(f)
            print("current path", os.path.dirname(os.path.realpath(__file__)))
            yaml_loc = current_wd + name_mapping[job]
            print("yaml location: ", yaml_loc)
        except:
            raise ValueError('Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.')
            status = 400
        print("Status:----> ", status)

        if status == 200:
            try:
                # setting environment variables:
                os.environ['APP_HOME'] = APP_HOME
            except FileNotFoundError:
                raise Exception("Environment variables are not set. Please set the variables!!")
        expt_config = parse_config(path=yaml_loc)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        updated_expt_name = expt_config['experiment_name'] + '_' + timestamp
        for k, v in expt_config["inputs"].items():  #handle input files from app data_input folder
            if expt_config["inputs"][k].startswith("inputs/"):
                expt_config["inputs"][k] = os.path.join(os.path.split(yaml_loc)[0], v)
        expt_config["experiment_name"] = updated_expt_name
        if "path_to_result_folder" in expt_config["inputs"]:
            expt_config["inputs"]["path_to_result_folder"] = os.path.join(
                expt_config["inputs"]["path_to_result_folder"], job, timestamp)
        directory = expt_config["inputs"]["path_to_result_folder"]
        if not os.path.exists(os.path.join(APP_HOME, job)):
            os.mkdir(os.path.join(APP_HOME, job))
        if not os.path.exists(directory):
            os.mkdir(directory)
        print("******DIRECTORY: ", directory)
        yaml_loc = os.path.join(directory, updated_expt_name + '.yaml')
        with open(yaml_loc, 'w') as f:
            yaml.dump(expt_config, f, default_flow_style=False)
        try:
            # Run daggit
            init_command = "daggit init " + yaml_loc
            subprocess.check_output(init_command, shell=True)
            run_command = "nohup daggit run " + updated_expt_name + " &"
            subprocess.check_output(run_command, shell=True)
            status = 200

        except:
            raise ValueError('DAG run fail. Experiment name: ' + updated_expt_name)
            status = 400
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
        response = jsonify(api_response)
        response.status_code = 200
        return  response
    else:
        response = jsonify(api_response)
        response.status_code = 400
        return  response

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
        status = 400
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
    response = jsonify(api_response)
    return  response

app.run(host='0.0.0.0', port=3579)
