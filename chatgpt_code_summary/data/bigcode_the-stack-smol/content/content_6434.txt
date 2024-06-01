# =============================================================================
#  periscope-ps (blipp)
#
#  Copyright (c) 2013-2016, Trustees of Indiana University,
#  All rights reserved.
#
#  This software may be modified and distributed under the terms of the BSD
#  license.  See the COPYING file for details.
#
#  This software was created at the Indiana University Center for Research in
#  Extreme Scale Technologies (CREST).
# =============================================================================
import time
from unis_client import UNISInstance
import settings
import pprint
from utils import merge_dicts
from requests.exceptions import ConnectionError

logger = settings.get_logger('conf')


class ServiceConfigure(object):
    '''
    ServiceConfigure is meant to be a generic class for any service
    which registers itself to, and gets configuration from UNIS. It
    was originally developed for BLiPP, but BLiPP specific features
    should be in the BlippConfigure class which extends
    ServiceConfigure.
    '''
    def __init__(self, initial_config={}, node_id=None, urn=None):
        if not node_id:
            node_id = settings.UNIS_ID
        self.node_id = node_id
        self.urn = urn
        self.config = initial_config
        self.unis = UNISInstance(self.config)
        self.node_setup = False
        self.service_setup = False
        self.exponential_backoff = int(self.config["properties"]["configurations"]["unis_poll_interval"])

    def initialize(self):
        try:
            r = self._setup_node(self.node_id)
            if not self.node_setup:
                return
            self._setup_service()
        except ConnectionError:
            return

    def refresh(self):
        try:
            r = self.unis.get("/services/" + self.config["id"])
            if not r:
                logger.warn('refresh', msg="refresh failed")
                logger.warn('refresh', msg="re-enable service")
                self._setup_service()
            else:
                self.config = r
                if time.time() * 1e+6 + int(self.config['properties']['configurations']['unis_poll_interval']) * 1e+6 >\
                    self.config['ts'] + self.config['ttl'] * 1e+6:
                    self._setup_service()
                    
            self.exponential_backoff = int(self.config['properties']['configurations']['unis_poll_interval'])
            return self.exponential_backoff
        except ConnectionError:
            self.exponential_backoff = self.exponential_backoff * 2
            return self.exponential_backoff

    def _setup_node(self, node_id):
        config = self.config
        props = self.config["properties"]["configurations"]
        logger.debug('_setup_node', config=pprint.pformat(config))
        hostname = settings.HOSTNAME
        urn = settings.HOST_URN if not self.urn else self.urn
        if node_id:
            r = self.unis.get("/nodes/" + str(node_id))
            if not r:
                logger.warn('_setup_node', msg="node id %s not found" % node_id)
                r = self.unis.post("/nodes",
                              data={
                        "$schema": settings.SCHEMAS["nodes"],
                        "name": hostname,
                        "urn": urn,
                        "id": node_id})
        if not node_id:
            r = self.unis.get("/nodes?urn=" + urn)
            if r and len(r):
                r = r[0]
                logger.info('_setup_node',
                            msg="Found node with our URN and id %s" % r["id"])
            else:
                r = self.unis.post("/nodes",
                              data={
                        "$schema": settings.SCHEMAS["nodes"],
                        "name": hostname,
                        "urn": urn})
            if r:
                self.node_id = r["id"]
        if r:
            if isinstance(r, list):
                r = r[0]
            config["runningOn"] = {
                "href": r["selfRef"],
                "rel": "full"}
            self.node_setup = True
        else:
            config["runningOn"] = {"href": ""}
            logger.warn('_setup_node', msg="Unable to set up BLiPP node in UNIS at %s" % props["unis_url"])

    def _setup_service(self):
        config = self.config
        props = self.config["properties"]["configurations"]
        logger.debug('_setup_service', config=pprint.pformat(config))
        r = None
        if config.get("id", None):
            r = self.unis.get("/services/" + config["id"])
        if not r:
            logger.warn('_setup_service',
                        msg="service id not specified or not found "\
                            "unis instance ...querying for service")
            rlist = self.unis.get("/services?name=" + config.get("name", None) +\
                                      "&runningOn.href=" + config["runningOn"]["href"] + "&limit=2")
            # loop over the returned services and find one that
            # doesn't return 410 see
            # https://uisapp2.iu.edu/jira-prd/browse/GEMINI-98
            if rlist:
                for i in range(len(rlist)):
                    r = self.unis.get('/services/' + rlist[i]["id"])
                    if r:
                        logger.info('_setup_service',
                                    msg="%s service found with id %s" % (config["name"], r["id"]))
                        break
            else:
                logger.warn('_setup_service',
                            msg="no service found by id or querying "\
                                "...creating new service at %s" % props["unis_url"])
            
        if isinstance(r, dict) and r:
            merge_dicts(config, r)

        # always update UNIS with the merged config
        r = None
        if config.get("id", None):
            r = self.unis.put("/services/" + config["id"], data=config)
        
        if not r:
            r = self.unis.post("/services", data=config)
        if r and isinstance(r, dict):
            merge_dicts(config, r)

        if r:
            self.service_setup = True
        else:
            logger.warn('_setup_service', msg="unable to set up service in UNIS")


    def get(self, key, default=None):
        try:
            return self.config[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        '''
        This allows an object which is an instance of this class to behave
        like a dictionary when queried with [] syntax
        '''
        return self.config[key]
