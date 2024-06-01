from data_common.config.configurer import get_conf
from data_common.provision.gs_buckets import confirm_bucket


def init_namespace_poc():

    conf = get_conf()

    project_id = conf.cloud.gcp.project
    namespaces = conf.namespaces

    for namespace, v in namespaces.items():

        print(f'namespace: {namespace}')
        bucket = confirm_bucket(
            bucket_name=namespace,
            project_id=project_id
        )

        print(bucket.name)
