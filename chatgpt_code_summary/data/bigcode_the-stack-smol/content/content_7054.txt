# celery settings

broker_url = 'amqp://guest:guest@rabbitmq:5672/'
result_backend = 'rpc://'
accept_content = ['json']
task_serializer = 'json'
task_soft_time_limit = 60 * 3  # 3 minute timeout
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

