"""
The script upadates variables in the inputs/credential.ini file. 
Running VDD Content tagging requires the following environment variables to be set:
GOOGLE_APPLICATION_CREDENTIALS - Path to google vision and speech api credentials
redis_host - Default localhost
redis_port - Default6379
redis_password - none
tagme_token - none
kafka_host - localhost
kafka_port - 9092
"""

import os
import configparser

#os.getenv('AIRFLOW_HOME')
base_path=os.path.dirname(os.path.realpath(__file__))
pathTocredentials = os.path.join(base_path,'inputs/credential_conf.ini')
config = configparser.ConfigParser(allow_no_value=True)
config.read(pathTocredentials)

try:
	config["google application credentials"]["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
except:
	print("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

try:
	config["redis"]["host"] = os.getenv('redis_host')	
except:
	print("redis_host environment variable not set. Defaulting to localhost. ")
	config["redis"]["host"] ="localhost"
try:
	config["redis"]["port"] = os.getenv('redis_port')	
except:
	print("redis_port environment variable not set. Defaulting to 6379.")
	config["redis"]["port"] = 6379
try:
	config["redis"]["password"] = os.getenv('redis_password')	
except:
	print("redis_password environment variable not set. Defaulting to none.")
	config["redis"]["password"] = "none"

try:
	config["tagme credentials"]["gcube_token"] = os.getenv('tagme_token')
except:
	print("gcube_token environment variable not set.Defaulting to none.")
	config["tagme credentials"]["gcube_token"] = "none"

try:
	config["kafka"]["host"] = os.getenv('kafka_host')
except:
	print("kafka_host environment variable not set. Defaulting to localhost. ")
	config["kafka"]["host"] = "localhost"
try:	
	config["kafka"]["port"] = os.getenv('kafka_port')
except:
	print("kafka_port environment variable not set. Defaulting to 9092.")
	config["kafka"]["port"] = 9092	

updatedPathTocredentials = os.path.join(base_path,'inputs/credentials.ini')
with open(updatedPathTocredentials, 'w+') as configfile:
	config.write(configfile)
