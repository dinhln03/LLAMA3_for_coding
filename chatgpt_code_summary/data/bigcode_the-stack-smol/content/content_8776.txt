from pathlib import Path
from datetime import datetime, timedelta
from src.settings import envs
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.hooks.postgres_hook import PostgresHook
import logging
from src.settings import log_config
import shutil

# Setting up module from __file__ as the interpreter sets __name__ as __main__ when the source file is executed as
# main program
logger = logging.getLogger(name=__file__.replace(envs.PROJECT_ROOT, '').replace('/', '.')[1:-3])

# these args will get passed on to each operator
# you can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(7),
    'email': ['airflow@airflow.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

DAG_ID = '{p.parent.name}_{p.stem}'.format(p=Path(__file__))
PARAMS = Variable.get(DAG_ID, deserialize_json=True)
SCHEDULE_INTERVAL = PARAMS.get('schedule_interval') or None
DAYS_TO_RETAIN = PARAMS.get('days_to_retain', 60)
TABLES = ("xcom", "task_instance", "sla_miss", "log", "dag_run", "task_fail", "task_reschedule")
LOG_DIR = '/usr/local/airflow/logs'

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='BT Bill Clean Up DAGs Metadata and Logs',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=1
)


def clean_dags_logs():
    hook = PostgresHook(postgres_conn_id="airflow_postgres")
    dag_files = Path(Path(__file__).parent).glob('*.py')
    dags = ['{p.parent.name}_{p.stem}'.format(p=p) for p in dag_files]
    dags = [d for d in dags if d != DAG_ID]

    execution_date = datetime.date(datetime.now()) - timedelta(days=DAYS_TO_RETAIN)
    p = Path(LOG_DIR)
    for d in dags:
        logger.info("Cleaning up meta tables for {}".format(d))
        for t in TABLES:
            sql = "delete from {} where dag_id='{}' and execution_date < '{}'".format(t, d, execution_date)
            hook.run(sql, True)
        logger.info('Cleaning up log folder for {}'.format(d))
        for path in list(p.glob('{}/*/*'.format(d))):
            log_date = str(path).split('/')[-1]
            log_date = log_date.split('T')[0]
            log_date = datetime.date(datetime.strptime(log_date, '%Y-%m-%d'))
            if log_date < execution_date:
                logger.info('Deleting dir {}'.format(str(path.absolute())))
                shutil.rmtree(str(path.absolute()))


clean_up = PythonOperator(
    task_id='clean_up',
    python_callable=clean_dags_logs,
    dag=dag)

dag >> clean_up
