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

# global declaration of api_response:
api_response = {
        "id": "api.daggit",
        "ver": "v1",
        "ts": "",
        "params": {
            "resmsgid": "null",
            "msgid": "",
            "err": "null",
            "status": "fail",
            "errmsg": "null"
        },
        "responseCode": "OK",
        "result": {
            "status": 400
        }
    }


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
    api_response["ts"] = time_format
    api_response["result"]["execution_date"] = date
    if request.is_json:
        try:
            req = request.get_json()
            try:
                APP_HOME = req["request"]["input"]["APP_HOME"]
                status = 200
            except KeyError:
                try:
                    APP_HOME = os.environ['APP_HOME']
                    status = 200
                except:
                    print("APP_HOME not set")
                    status = 400
            
            try:
                job = req["request"]["job"]
                status = 200
            except KeyError:
                print("job required to initiate experiment.")
                status = 400
            
        except ValueError:
            status = 400

        try:
            with open(os.path.join(current_wd, 'expt_name_map.json')) as f:
                name_mapping = json.load(f)
            print("current path", os.path.dirname(os.path.realpath(__file__)))
            yaml_loc = current_wd + name_mapping[job]
            print("yaml location: ", yaml_loc)
        except ValueError:
            status = 400
            raise Exception('Unrecognised experiment_name. Check expt_name_map.json for recognised experiment name.')

        if status == 200:
            os.environ['APP_HOME'] = APP_HOME
        
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
            output_directory = expt_config["inputs"]["path_to_result_folder"]
        else:
            output_directory = os.path.join(APP_HOME, job, timestamp)
        

        if not os.path.exists(os.path.join(APP_HOME, job)):
            os.mkdir(os.path.join(APP_HOME, job))
            print("making "+ os.path.join(APP_HOME, job))
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
            print("making "+ output_directory)

        #dump input passed in api and write to yaml
        try:
            dag_input_path = os.path.join(output_directory, 'input')
            if not os.path.exists(dag_input_path):
                        os.mkdir(dag_input_path)
                        print("making "+dag_input_path)

            argcount = 0
            for submit_input in req["request"]["input"]:
                if submit_input not in["APP_HOME"]:
                    input_data = {submit_input:req["request"]["input"][submit_input]}
                    argcount+=1
                    with open(os.path.join(dag_input_path ,str(submit_input)+'.json'), 'w') as outfile:
                        json.dump(input_data, outfile)
                    try:
                        expt_config["inputs"]["input_arg"+str(argcount)] = os.path.join(dag_input_path ,str(submit_input)+'.json')
                    except:
                        print("Passed argument not updated in yaml. 'input_arg"+str(argcount)+"' field not found in inputs.")
        except:
            pass

        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                            filename=os.path.join(output_directory, "daggit_api.log"))
        logger = logging.getLogger(__name__)
        print("******DIRECTORY: ", output_directory)
        yaml_loc = os.path.join(output_directory, updated_expt_name + '.yaml')
        
        with open(yaml_loc, 'w') as f:
            yaml.dump(expt_config, f, default_flow_style=False)
        init_command = "daggit init " + yaml_loc
        out_init, res_init = subprocess.getstatusoutput(init_command)
        
        if out_init == 0:
            logging.info("----- Daggit Initialization successful :) Starting run.-----")
            run_command = "nohup daggit run " + updated_expt_name
            status=200
            api_response["params"]["status"] = "success"

            #out_run, res_run = subprocess.getstatusoutput(run_command)
            out_run = subprocess.Popen(run_command.split()).pid

            #with subprocess.Popen(run_command.split(), stdout=subprocess.PIPE) as proc:
            #    logger.write(proc.stdout.read())

            """
            out_run, res_run = subprocess.Popen(run_command.split())
            if out_run != 0:
                logging.info("----- DAG run failed for experiment: {0} -----".format(updated_expt_name), exc_info=True)
                logger.error(res_run, exc_info=True)
                raise ValueError(res_run)
            else:
                logging.info("----- Daggit Run Successful :) -----", exc_info=True)
                status = 200
            """
        else:
            logging.info("----- Unsuccessful Daggit Initialization -----", exc_info=True)
            logger.error(res_init, exc_info=True)
            status = 400
            api_response["params"]["status"] = "Fail"
            api_response["params"]["errmsg"] = "Dag Initialization failed"
            raise FileNotFoundError(res_init)

        api_response["result"]["status"] = status
        api_response["result"]["experiment_name"] = updated_expt_name
        # api_response["result"]["execution_date"] = date
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
    except ValueError:
        return Response(status=400)
    print("*****expt_name:", expt_name)
    try:
        from_time = request.args.get('execution_date')
        command = "airflow dag_state " + expt_name + " " + from_time
        status = subprocess.check_output(command, shell=True)
    except ValueError:
        status = 400
        print("Execution date invalid")
    time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
    api_response["ts"] = time_format
    response = jsonify(api_response)
    return response


@app.route('/daggit/status', methods=['POST'])
def get_dag_status_from_log():
    time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
    api_response["ts"] = time_format
    if request.is_json:
        try:
            req = request.get_json()
            EXPERIMENT_HOME = req["request"]["input"]["EXPERIMENT_HOME"]
            print("******EXPERIMENT_HOME: ", EXPERIMENT_HOME)
            status = 200
        except ValueError:
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
