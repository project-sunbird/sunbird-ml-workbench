import os
import requests
import time
from bert_serving.client import BertClient
from flask import Flask
from flask import request, Response,jsonify
from daggit.core.io.io import KafkaDispatcher, KafkaCLI
import configparser
import logging


app = Flask(__name__)

DS_DATA_HOME = os.environ['DS_DATA_HOME']
base_path=os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_path,"apps/vectorise_service/inputs/multi_cased_L-12_H-768_A-12")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                            filename=os.path.join(base_path, "vectorService.log"))
logger = logging.getLogger(__name__)


logging.info("Loading model from ", model_path)

logging.info("BERT server started at port:5555. Ready to serve vectors!" )
logging.info("Rendering /ml/vector/search and /vector/ContentText apis at port:1729. ")
base_path=os.path.dirname(os.path.realpath(__file__))
pathTocredentials = os.path.join(base_path,'inputs/credentials.ini')
config = configparser.ConfigParser(allow_no_value=True)
config.read(pathTocredentials)

api_response = {
    "id": "api.ml.vector",
    "ets": "",
    "params": {
            "resmsgid": "null",
            "msgid": "",
            "err": "null",
            "status": "fail",
            "errmsg": "null"
        },
    "result":{
        "action":"get_BERT_embedding",
        "vector": ""        
    }
 }

@app.route('/ml/vector/search', methods=['POST'])
def getTextVec():

	if request.is_json:
		try:
			req = request.get_json()
			try:
				time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
				api_response["ets"] = time_format
				if req["request"]['method']=="BERT" and req["request"]["language"]=="en":
					text = req["request"]["text"]
					app.logger.info("trying to connect to BERT Client")
					bc = BertClient(timeout = 2000)
					vector = bc.encode(text)
					bc.close()
					api_response["result"]["vector"]= vector.tolist()
					api_response["params"]["status"]= "success"
					status=200
					logging.info("here")
					response = jsonify(api_response)
					response.status_code = 200
					return response

			except ValueError:
				api_response["params"]["errmsg"] = "Bert service not available."
				api_response["params"]["status"]= "fail"
				response = jsonify(api_response)
				response.status_code = 400
				return response
		except:
			api_response["params"]["errmsg"] = "API request in incorrect format."
			api_response["params"]["status"]= "fail"
			response = jsonify(api_response)
			response.status_code = 400
			return response

	else:
		response = jsonify(api_response)
		response.status_code = 500
		return response

def writeToKafka(pathTocredentials, vector, cid, topic_name):

	time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
	topic_output = {
	    "eid": "MVC_JOB_PROCESSOR",
	    "ets": time_format,
	    "mid": "LP.1591603456223.a5d1c6f0-a95e-11ea-b80d-b75468d19fe4",
	    "actor": {
	      "id": "UPDATE ML CONTENT TEXT VECTOR",
	      "type": "System"
	    },
	    "context": {
	      "pdata": {
	        "ver": "1.0",
	        "id": "org.ekstep.platform"
	      },
	      "channel": "01285019302823526477"
	    },
	    "object": {
	      "ver": "1.0",
	      "id": cid
	    },
	    "eventData":{
	        "action": "update-ml-contenttextvector",
	        "stage": 3,
	        "ml_contentTextVector":vector
	    }
	}
	app.logger.info("Trying to write "+str(topic_output)+" to "+str(topic_name))
	kafka_cli = KafkaCLI(pathTocredentials)
	status = kafka_cli.write(topic_output, topic_name)
	if status:
		app.logger.info("Write to kafka successful.")
	else:
		app.logger.info("Failed to write to kafka")
	return status

@app.route('/ml/vector/ContentText', methods=['POST'])
def getContentTextVDD():
	time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
	api_response["ets"] = time_format
	app.logger.info("/vector/ContentText called.")
	try:
		request.is_json
	except:
		app.logger.info("request wrong format")

	if request.is_json:
		try:
			req = request.get_json()
			app.logger.info("/vector/ContentText called.")
			try:
				if req["request"]['method']=="BERT" and req["request"]["language"]=="en":
					app.logger.info("Vectorisation method: BERT english ")
					text = req["request"]["text"]
					cid = req["request"]["cid"]
					topic_name = config['kafka']['topic_name']
					bc = BertClient(timeout = 2000)
					vector = bc.encode(text)
					bc.close()
					vector_list = vector.tolist()
					#vector_list =[]
					api_response["result"]["vector"]= vector_list
					#print(api_response)
					app.logger.info("Initiating write to kafka.")

					kafka_write_status = writeToKafka(pathTocredentials, vector_list, cid, topic_name)
					if kafka_write_status:
						api_response["params"]["status"]= "success"
						status=200
						app.logger.info("Success writing to kafka")
					else:
						api_response["params"]["status"]= "fail"
						status=400

					response = jsonify(api_response)
					response.status_code = status
					return response
				else:
					app.logger.info("unidentified model or language parameter")
					status=400
					response = jsonify(api_response)
					response.status_code = status
					return response

			except ValueError:
				api_response["params"]["status"]= "fail"
				response = jsonify(api_response)
				response.status_code = 400
				return response
		except:
			api_response["params"]["status"]= "fail"
			response = jsonify(api_response)
			response.status_code = 400
			return response

	else:
		response = jsonify(api_response)
		response.status_code = 500
		return response

app.run(host='0.0.0.0', port=1729,threaded=True )
#server.close()

