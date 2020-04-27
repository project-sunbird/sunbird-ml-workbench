import json
import os
import subprocess
import time
import logging
import re

import yaml
from daggit.core.base.utils import parse_config
from flask import Flask
from flask import request, Response,jsonify

app = Flask(__name__)


def match_date(line, level_tag):
    match_this = ""
    matched = re.search(level_tag, line)
    if matched:
        # matches a date and adds it to matchThis
        match_this = matched.group()
    else:
        match_this = "NONE"
    return match_this


def get_response(log_fh, api_response, level_tag):
    for line in log_fh:
        if match_date(line, level_tag) != "NONE":
            api_response["result"]["status"] = "fail"
            api_response["result"]["status"] = 400
            return api_response
        else:
            continue
    api_response["result"]["status"] = 200
    api_response["params"]["status"] = "success"
    return api_response


@app.route('/daggit/submit', methods=['POST'])
def submit_dag():
    current_wd = os.path.dirname(os.path.realpath(__file__))
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
            "status": "fail",
            "errmsg": "null"
        },
        "responseCode": "OK",
        "result": {
            "status": status,
            "experiment_name": "null",
            "estimate_time": "",
            "execution_date": date
        }
    }
    if (request.is_json):
        try:
            req = request.get_json()
            APP_HOME = req["request"]["input"]["APP_HOME"]
            job = req["request"]["job"]
            print("******JOB: ", job)
            status = 200
        except:
            status = 400

        try:
            with open(os.path.join(current_wd, 'expt_name_map.json')) as f:
                name_mapping = json.load(f)
            print("current path", os.path.dirname(os.path.realpath(__file__)))
            yaml_loc = current_wd + name_mapping[job]
            print("yaml location: ", yaml_loc)
        except:
            status = 400
            raise ValueError('Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.')

        if status == 200:
            try:
                # setting environment variables:
                os.environ['APP_HOME'] = APP_HOME
            except FileNotFoundError:
                raise Exception("Environment variables are not set. Please set the variables!!")
        expt_config = parse_config(path=yaml_loc)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        updated_expt_name = expt_config['experiment_name'] + '_' + timestamp
        for k, v in expt_config["inputs"].items():  # handle input files from app data_input folder
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
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                            filename=os.path.join(directory, "daggit_api.log"))
        logger = logging.getLogger(__name__)
        print("******DIRECTORY: ", directory)
        yaml_loc = os.path.join(directory, updated_expt_name + '.yaml')
        with open(yaml_loc, 'w') as f:
            yaml.dump(expt_config, f, default_flow_style=False)
        init_command = "daggit init " + yaml_loc
        out_init, res_init = subprocess.getstatusoutput(init_command)
        if out_init == 0:
            logging.info("----- Daggit Initialization successful :) -----")
            print()
            print()
            run_command = "daggit run " + updated_expt_name
            out_run, res_run = subprocess.getstatusoutput(run_command)
            if out_run != 0:
                logging.info("----- DAG run failed for experiment: {0} -----".format(updated_expt_name), exc_info=True)
                logger.error(res_run, exc_info=True)
                raise ValueError(res_run)
            else:
                logging.info("----- Daggit Run Successful :) -----", exc_info=True)
                status = 200
        else:
            logging.info("----- Unsuccessful Daggit Initialization -----", exc_info=True)
            logger.error(res_init, exc_info=True)
            raise FileNotFoundError(res_init)

        api_response["params"]["status"] = "success"
        api_response["result"]["status"] = status
        api_response["result"]["experiment_name"] = updated_expt_name
        api_response["result"]["execution_date"] = date
        logging.info("API Status: {0}".format(api_response["params"]["status"]))
        logging.info("Experiment name: {0}".format(api_response["result"]["experiment_name"]))
        logging.info("Execution date: {0}".format(api_response["result"]["execution_date"]))
        response = jsonify(api_response)
        response.status_code = 200
        return response
    else:
        response = jsonify(api_response)
        response.status_code = 400
        return response
    logging.info("API Status: ", api_response["params"]["status"])
    logging.info("Experiment name: ", api_response["result"]["experiment_name"])
    logging.info("Execution date: ", api_response["result"]["execution_date"])


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
    return response


@app.route('/daggit/status', methods=['POST'])
def get_dag_status_from_log():
    time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
    api_response = {
        "id": "api.daggit.status",
        "ver": "v1",
        "ts": time_format,
        "params": {
            "resmsgid": "null",
            "msgid": "",
            "err": "null",
            "status": "fail",
            "errmsg": "null"
        },
        "responseCode": "OK",
        "result": {
            "status": 400,
        }
    }

    if request.is_json:
        try:
            req = request.get_json()
            EXPERIMENT_HOME = req["request"]["input"]["EXPERIMENT_HOME"]
            print("******EXPERIMENT_HOME: ", EXPERIMENT_HOME)
            status = 200
        except:
            status = 400
        if "daggit_api.log" in [os.path.split(file)[1] for file in os.listdir(EXPERIMENT_HOME)]:
            with open(os.path.join(EXPERIMENT_HOME, "daggit_api.log"), "r") as log_file:
                result = get_response(log_file, api_response, "ERROR")
                response = jsonify(result)
                return response
        else:
            response = jsonify(api_response)
            response.status_code = 400
            return response
    else:
        response = jsonify(api_response)
        response.status_code = 400
        return response


app.run(host='0.0.0.0', port=3579)
