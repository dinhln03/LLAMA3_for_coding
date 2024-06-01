import os
from keystoneauth1.identity import v3
from keystoneauth1 import session
from novaclient import client

VERSION=2
AUTH_URL=os.getenv("OS_AUTH_URL")
USERNAME=os.getenv("OS_USERNAME")
PASSWORD=os.getenv("OS_PASSWORD")
PROJECT_ID=os.getenv("OS_PROJECT_ID")
PROJECT_NAME=os.getenv("OS_PROJECT_NAME")
USER_DOMAIN_ID=os.getenv("OS_USER_DOMAIN_ID")
PROJECT_DOMAIN_ID=os.getenv("OS_PROJECT_DOMAIN_ID")
CACERT=os.getenv("OS_CACERT")

def auth():
    auth = v3.Password(auth_url=AUTH_URL, username=USERNAME, password=PASSWORD,
            project_name=PROJECT_NAME, user_domain_id=USER_DOMAIN_ID,
            project_domain_id=PROJECT_DOMAIN_ID)

    sess = session.Session(auth=auth,verify=CACERT)
    nova  = client.Client(VERSION, session=sess)
    return nova
