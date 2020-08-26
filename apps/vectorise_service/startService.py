import os
import subprocess
import sys
import ssl
import logging



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


base_path=os.path.dirname(os.path.realpath(__file__))
daggit_home="/".join(base_path.split("/")[:-2])


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                            filename=os.path.join(base_path, "startService.log"))
logger = logging.getLogger(__name__)


subprocess.check_call([sys.executable, "-m", "pip", "install","-r",base_path+"/requirement.txt"])
logging.info("daggit home:%s ", daggit_home)
#subprocess.check_call([sys.executable, "-m", "pip", "install",daggit_home+"/bin/daggit-0.5.0.tar.gz"])


import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read(os.path.join(base_path,"inputs/credentials.ini"))
try:	
	config["kafka"]["topic_name"] = ".".join([os.getenv('env'),"mvc.processor.job.request"])
except:
	logging.info("env environment variable not set. Defaulting to sunbirddock.")
	config["kafka"]["topic_name"] = "sunbirddock.mvc.processor.job.request"
updatedPathTocredentials = os.path.join(base_path,'inputs/credentials.ini')

with open(updatedPathTocredentials, 'w+') as configfile:
	config.write(configfile)


### Get the model
from daggit.core.io.files import downloadZipFile
model_url = config["pretrained BERT"]["path"]
model_name = os.path.split(model_url)[1][:-4]
model_dir = os.path.join(daggit_home,"apps/vectorise_service/inputs", model_name)


if not os.path.isdir(model_dir):
	logging.info("Downloading vectorisation model.")
	downloadZipFile(model_url, os.path.join(daggit_home,"apps/vectorise_service/inputs"))
else:
	logging.info("Using "+model_name+" in "+model_dir+". To download a different model, remove the folder.")

### Start service

start_command = "bert-serving-start -model_dir "+model_dir +"/ -num_worker=1 "
logging.info(start_command)
#os.system(start_command)

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient

base_path=os.path.dirname(os.path.realpath(__file__))

args = get_args_parser().parse_args(['-model_dir', model_dir,
                                     '-port', '5555',
                                     '-port_out', '5556',
                                     '-max_seq_len', '40',
                                     '-mask_cls_sep'])
server = BertServer(args)
server.start()

logging.info("BERT server started at port:5555. Ready to serve vectors!" )

#start service
os.system("nohup python "+os.path.join(base_path,'vectorise_api.py'))

server.close()
logging.info("server closed")