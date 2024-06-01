# -*- coding: utf-8 -*-
#---------------------------------------------------------------------------
# Copyright 2020 VMware, Inc.  All rights reserved.

# AUTO GENERATED FILE -- DO NOT MODIFY!
#
# vAPI stub file for package com.vmware.vmc.orgs.sddcs.networks.edges.firewall.
#---------------------------------------------------------------------------

"""


"""

__author__ = 'VMware, Inc.'
__docformat__ = 'restructuredtext en'

import sys

from vmware.vapi.bindings import type
from vmware.vapi.bindings.converter import TypeConverter
from vmware.vapi.bindings.enum import Enum
from vmware.vapi.bindings.error import VapiError
from vmware.vapi.bindings.struct import VapiStruct
from vmware.vapi.bindings.stub import (
    ApiInterfaceStub, StubFactoryBase, VapiInterface)
from vmware.vapi.bindings.common import raise_core_exception
from vmware.vapi.data.validator import (UnionValidator, HasFieldsOfValidator)
from vmware.vapi.exception import CoreException
from vmware.vapi.lib.constants import TaskType
from vmware.vapi.lib.rest import OperationRestMetadata


class Config(VapiInterface):
    """
    
    """

    _VAPI_SERVICE_ID = 'com.vmware.vmc.orgs.sddcs.networks.edges.firewall.config'
    """
    Identifier of the service in canonical form.
    """
    def __init__(self, config):
        """
        :type  config: :class:`vmware.vapi.bindings.stub.StubConfiguration`
        :param config: Configuration to be used for creating the stub.
        """
        VapiInterface.__init__(self, config, _ConfigStub)
        self._VAPI_OPERATION_IDS = {}


    def delete(self,
               org,
               sddc,
               edge_id,
               ):
        """
        Delete firewall configuration for a management or compute gateway (NSX
        Edge).

        :type  org: :class:`str`
        :param org: Organization identifier. (required)
        :type  sddc: :class:`str`
        :param sddc: Sddc Identifier. (required)
        :type  edge_id: :class:`str`
        :param edge_id: Edge Identifier. (required)
        :raise: :class:`com.vmware.vapi.std.errors_client.InvalidRequest` 
             Bad request. Request object passed is invalid.
        :raise: :class:`com.vmware.vapi.std.errors_client.Unauthorized` 
             Forbidden. Authorization header not provided.
        :raise: :class:`com.vmware.vapi.std.errors_client.NotFound` 
             Not found. Requested object not found.
        """
        return self._invoke('delete',
                            {
                            'org': org,
                            'sddc': sddc,
                            'edge_id': edge_id,
                            })

    def get(self,
            org,
            sddc,
            edge_id,
            ):
        """
        Retrieve the firewall configuration for a management or compute gateway
        (NSX Edge).

        :type  org: :class:`str`
        :param org: Organization identifier. (required)
        :type  sddc: :class:`str`
        :param sddc: Sddc Identifier. (required)
        :type  edge_id: :class:`str`
        :param edge_id: Edge Identifier. (required)
        :rtype: :class:`com.vmware.vmc.model_client.FirewallConfig`
        :return: com.vmware.vmc.model.FirewallConfig
        :raise: :class:`com.vmware.vapi.std.errors_client.InvalidRequest` 
             Bad request. Request object passed is invalid.
        :raise: :class:`com.vmware.vapi.std.errors_client.Unauthorized` 
             Forbidden. Authorization header not provided.
        :raise: :class:`com.vmware.vapi.std.errors_client.NotFound` 
             Not found. Requested object not found.
        """
        return self._invoke('get',
                            {
                            'org': org,
                            'sddc': sddc,
                            'edge_id': edge_id,
                            })

    def update(self,
               org,
               sddc,
               edge_id,
               firewall_config,
               ):
        """
        Configure firewall for a management or compute gateway (NSX Edge).

        :type  org: :class:`str`
        :param org: Organization identifier. (required)
        :type  sddc: :class:`str`
        :param sddc: Sddc Identifier. (required)
        :type  edge_id: :class:`str`
        :param edge_id: Edge Identifier. (required)
        :type  firewall_config: :class:`com.vmware.vmc.model_client.FirewallConfig`
        :param firewall_config: (required)
        :raise: :class:`com.vmware.vapi.std.errors_client.InvalidRequest` 
             Bad request. Request object passed is invalid.
        :raise: :class:`com.vmware.vapi.std.errors_client.Unauthorized` 
             Forbidden. Authorization header not provided.
        :raise: :class:`com.vmware.vapi.std.errors_client.NotFound` 
             Not found. Requested object not found.
        """
        return self._invoke('update',
                            {
                            'org': org,
                            'sddc': sddc,
                            'edge_id': edge_id,
                            'firewall_config': firewall_config,
                            })
class Statistics(VapiInterface):
    """
    
    """

    _VAPI_SERVICE_ID = 'com.vmware.vmc.orgs.sddcs.networks.edges.firewall.statistics'
    """
    Identifier of the service in canonical form.
    """
    def __init__(self, config):
        """
        :type  config: :class:`vmware.vapi.bindings.stub.StubConfiguration`
        :param config: Configuration to be used for creating the stub.
        """
        VapiInterface.__init__(self, config, _StatisticsStub)
        self._VAPI_OPERATION_IDS = {}


    def get(self,
            org,
            sddc,
            edge_id,
            rule_id,
            ):
        """
        Retrieve statistics for a specific firewall rule for a management or
        compute gateway (NSX Edge).

        :type  org: :class:`str`
        :param org: Organization identifier. (required)
        :type  sddc: :class:`str`
        :param sddc: Sddc Identifier. (required)
        :type  edge_id: :class:`str`
        :param edge_id: Edge Identifier. (required)
        :type  rule_id: :class:`long`
        :param rule_id: Rule Identifier. (required)
        :rtype: :class:`com.vmware.vmc.model_client.FirewallRuleStats`
        :return: com.vmware.vmc.model.FirewallRuleStats
        :raise: :class:`com.vmware.vapi.std.errors_client.InvalidRequest` 
             Bad request. Request object passed is invalid.
        :raise: :class:`com.vmware.vapi.std.errors_client.Unauthorized` 
             Forbidden. Authorization header not provided
        :raise: :class:`com.vmware.vapi.std.errors_client.NotFound` 
             Not found. Requested object not found.
        """
        return self._invoke('get',
                            {
                            'org': org,
                            'sddc': sddc,
                            'edge_id': edge_id,
                            'rule_id': rule_id,
                            })
class _ConfigStub(ApiInterfaceStub):
    def __init__(self, config):
        # properties for delete operation
        delete_input_type = type.StructType('operation-input', {
            'org': type.StringType(),
            'sddc': type.StringType(),
            'edge_id': type.StringType(),
        })
        delete_error_dict = {
            'com.vmware.vapi.std.errors.invalid_request':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'InvalidRequest'),
            'com.vmware.vapi.std.errors.unauthorized':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'Unauthorized'),
            'com.vmware.vapi.std.errors.not_found':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'NotFound'),

        }
        delete_input_value_validator_list = [
        ]
        delete_output_validator_list = [
        ]
        delete_rest_metadata = OperationRestMetadata(
            http_method='DELETE',
            url_template='/vmc/api/orgs/{org}/sddcs/{sddc}/networks/4.0/edges/{edgeId}/firewall/config',
            path_variables={
                'org': 'org',
                'sddc': 'sddc',
                'edge_id': 'edgeId',
            },
            query_parameters={
            },
            content_type='application/json'
        )

        # properties for get operation
        get_input_type = type.StructType('operation-input', {
            'org': type.StringType(),
            'sddc': type.StringType(),
            'edge_id': type.StringType(),
        })
        get_error_dict = {
            'com.vmware.vapi.std.errors.invalid_request':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'InvalidRequest'),
            'com.vmware.vapi.std.errors.unauthorized':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'Unauthorized'),
            'com.vmware.vapi.std.errors.not_found':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'NotFound'),

        }
        get_input_value_validator_list = [
        ]
        get_output_validator_list = [
        ]
        get_rest_metadata = OperationRestMetadata(
            http_method='GET',
            url_template='/vmc/api/orgs/{org}/sddcs/{sddc}/networks/4.0/edges/{edgeId}/firewall/config',
            path_variables={
                'org': 'org',
                'sddc': 'sddc',
                'edge_id': 'edgeId',
            },
            query_parameters={
            },
            content_type='application/json'
        )

        # properties for update operation
        update_input_type = type.StructType('operation-input', {
            'org': type.StringType(),
            'sddc': type.StringType(),
            'edge_id': type.StringType(),
            'firewall_config': type.ReferenceType('com.vmware.vmc.model_client', 'FirewallConfig'),
        })
        update_error_dict = {
            'com.vmware.vapi.std.errors.invalid_request':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'InvalidRequest'),
            'com.vmware.vapi.std.errors.unauthorized':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'Unauthorized'),
            'com.vmware.vapi.std.errors.not_found':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'NotFound'),

        }
        update_input_value_validator_list = [
        ]
        update_output_validator_list = [
        ]
        update_rest_metadata = OperationRestMetadata(
            http_method='PUT',
            url_template='/vmc/api/orgs/{org}/sddcs/{sddc}/networks/4.0/edges/{edgeId}/firewall/config',
            request_body_parameter='firewall_config',
            path_variables={
                'org': 'org',
                'sddc': 'sddc',
                'edge_id': 'edgeId',
            },
            query_parameters={
            },
            content_type='application/json'
        )

        operations = {
            'delete': {
                'input_type': delete_input_type,
                'output_type': type.VoidType(),
                'errors': delete_error_dict,
                'input_value_validator_list': delete_input_value_validator_list,
                'output_validator_list': delete_output_validator_list,
                'task_type': TaskType.NONE,
            },
            'get': {
                'input_type': get_input_type,
                'output_type': type.ReferenceType('com.vmware.vmc.model_client', 'FirewallConfig'),
                'errors': get_error_dict,
                'input_value_validator_list': get_input_value_validator_list,
                'output_validator_list': get_output_validator_list,
                'task_type': TaskType.NONE,
            },
            'update': {
                'input_type': update_input_type,
                'output_type': type.VoidType(),
                'errors': update_error_dict,
                'input_value_validator_list': update_input_value_validator_list,
                'output_validator_list': update_output_validator_list,
                'task_type': TaskType.NONE,
            },
        }
        rest_metadata = {
            'delete': delete_rest_metadata,
            'get': get_rest_metadata,
            'update': update_rest_metadata,
        }
        ApiInterfaceStub.__init__(
            self, iface_name='com.vmware.vmc.orgs.sddcs.networks.edges.firewall.config',
            config=config, operations=operations, rest_metadata=rest_metadata,
            is_vapi_rest=False)

class _StatisticsStub(ApiInterfaceStub):
    def __init__(self, config):
        # properties for get operation
        get_input_type = type.StructType('operation-input', {
            'org': type.StringType(),
            'sddc': type.StringType(),
            'edge_id': type.StringType(),
            'rule_id': type.IntegerType(),
        })
        get_error_dict = {
            'com.vmware.vapi.std.errors.invalid_request':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'InvalidRequest'),
            'com.vmware.vapi.std.errors.unauthorized':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'Unauthorized'),
            'com.vmware.vapi.std.errors.not_found':
                type.ReferenceType('com.vmware.vapi.std.errors_client', 'NotFound'),

        }
        get_input_value_validator_list = [
        ]
        get_output_validator_list = [
        ]
        get_rest_metadata = OperationRestMetadata(
            http_method='GET',
            url_template='/vmc/api/orgs/{org}/sddcs/{sddc}/networks/4.0/edges/{edgeId}/firewall/statistics/{ruleId}',
            path_variables={
                'org': 'org',
                'sddc': 'sddc',
                'edge_id': 'edgeId',
                'rule_id': 'ruleId',
            },
            query_parameters={
            },
            content_type='application/json'
        )

        operations = {
            'get': {
                'input_type': get_input_type,
                'output_type': type.ReferenceType('com.vmware.vmc.model_client', 'FirewallRuleStats'),
                'errors': get_error_dict,
                'input_value_validator_list': get_input_value_validator_list,
                'output_validator_list': get_output_validator_list,
                'task_type': TaskType.NONE,
            },
        }
        rest_metadata = {
            'get': get_rest_metadata,
        }
        ApiInterfaceStub.__init__(
            self, iface_name='com.vmware.vmc.orgs.sddcs.networks.edges.firewall.statistics',
            config=config, operations=operations, rest_metadata=rest_metadata,
            is_vapi_rest=False)


class StubFactory(StubFactoryBase):
    _attrs = {
        'Config': Config,
        'Statistics': Statistics,
        'config': 'com.vmware.vmc.orgs.sddcs.networks.edges.firewall.config_client.StubFactory',
    }

