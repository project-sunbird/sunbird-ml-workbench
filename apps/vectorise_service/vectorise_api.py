import os
import requests
import time
from bert_serving.client import BertClient
from flask import Flask
from flask import request, Response,jsonify
from daggit.core.io.io import KafkaDispatcher, KafkaCLI
import configparser


app = Flask(__name__)

DS_DATA_HOME = os.environ['DS_DATA_HOME']
model_path = os.path.join(DS_DATA_HOME+"BERT_models/multi_cased_L-12_H-768_A-12")
base_path=os.path.dirname(os.path.realpath(__file__))

print("BERT server started at port:5555. Ready to serve vectors!" )
print("Rendering /ml/vector/search and /vector/ContentText apis at port:1729. ")
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
				if req['model']=="BERT" & req["language"]=="en":
					text = req["text"]
					bc = BertClient()
					vector = bc.encode(text)
					bc.close()
					api_response["result"]["vector"]= vector.tolist()
					api_response["params"]["status"]= "success"
					status=200
					time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
					api_response["ets"] = time_format

					response = jsonify(api_response)
					response.status_code = 200
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
	    "edata":{
	        "action": "update-ml-contenttextvector",
	        "stage": 3,
	        "ml_contentTextVector":vector
	    }
	}
	print("Trying to write "+str(topic_output)+" to "+str(topic_name))
	kafka_cli = KafkaCLI(pathTocredentials)
	status = kafka_cli.write(topic_output, topic_name)
	if status:
		print("Write to kafka successful.")
	else:
		print("Failed to write to kafka")
	return status

@app.route('/ml/vector/ContentText', methods=['POST'])
def getContentTextVDD():
	time_format = time.strftime("%Y-%m-%d %H:%M:%S:%s")
	api_response["ets"] = time_format

	if request.is_json:
		try:
			req = request.get_json()
			print("/vector/ContentText called.")
			try:
				if req["request"]['method']=="BERT" and req["request"]["language"]=="en":
					text = req["request"]["text"]
					cid = req["request"]["cid"]
					topic_name = config['kafka']['topic_name']
					bc = BertClient()
					vector = bc.encode(text)
					bc.close()
					vector_list = vector.tolist()
					#vector_list =[]
					api_response["result"]["vector"]= vector_list
					print(api_response)
					kafka_write_status = writeToKafka(pathTocredentials, vector_list, cid, topic_name)
					if kafka_write_status:
						api_response["params"]["status"]= "success"
						status=200
						print("Success writing to kafka")
					else:
						api_response["params"]["status"]= "fail"
						status=400

					response = jsonify(api_response)
					response.status_code = status
					return response
				else:
					print("unidentified model or language parameter")
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

app.run(host='0.0.0.0', port=1729)
#server.close()

