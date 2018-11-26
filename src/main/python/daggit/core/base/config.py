DAGGIT_HOME = "DAGGIT_HOME"
STORE = 'Local'
ORCHESTRATOR = 'airflow'
STORAGE_FORMAT = '.h5'
ORCHESTRATOR_ATTR = dict({'AIRFLOW': dict({'dag_config':dict({'depends_on_past': False, 'schedule_interval':'@once','start_date':'20-11-2018'})})})