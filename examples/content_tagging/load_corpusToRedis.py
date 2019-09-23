import os
import redis
import configparser
import pandas as pd

credentials_loc = os.path.join(os.getcwd(), "examples/content_tagging/inputs/credentials.ini")
config = configparser.ConfigParser(allow_no_value=True)
config.read(credentials_loc)
redis_host = config["redis"]["host"]
redis_port = config["redis"]["port"]
redis_password = config["redis"]["password"]

loc=os.path.join(os.getcwd(), "examples/content_tagging/inputs/corpus")
files=os.listdir(loc)

r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
for f in files:
    df=pd.read_csv(os.path.join(loc,f))
    pipe = r.pipeline()
    for ind,val in df.iterrows():
        pipe.set(f[:-4]+'.'+val['keyword'], val['dbpedia_score'])
    pipe.execute()