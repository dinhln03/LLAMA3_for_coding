# -*- coding:utf-8 -*-
"""
    Copyright (c) 2013-2016 SYPH, All Rights Reserved.
    -----------------------------------------------------------
    Author: S.JunPeng
    Date:  2016/12/22
    Change Activity:
"""
import logging
import json

from vendor.utils.encrypt import Cryption
from apps.common.models import ClientOverview
from apps.remote.models import FeatureFieldRel
from apps.etl.context import ApplyContext
from vendor.errors.api_errors import *

logger = logging.getLogger('apps.featureapi')


class Judger(object):
    """
        1.authentication (_check_identity)
        2.data decryption (_decrypt)
        3.check availability of arguments (_args_useful_check)
        4.throw the Exceptions
        5.finally check all works
    """

    def __init__(self, client_code, data):
        self.client_code = client_code
        self.client_id = ''
        self.client_secret = ''
        self.des_key = ''
        self.origin_data = data
        self.cryption = Cryption()
        self.apply_id = ''
        self.target_features = []
        self.arguments = {}
        self.ret_msg = []

    def _check_sum(self):
        if self.client_id and self.client_secret and self.des_key and self.target_features and self.arguments \
                and (len(self.target_features) == len(self.ret_msg)):
            return True
        else:
            return False

    def _check_identity(self):
        client_package = ClientOverview.objects.filter(client_code=self.client_code)
        if not client_package:
            logger.error('Response from the function of `judge._check_identity`, error_msg=%s, rel_err_msg=%s'
                         % (UserIdentityError.message, 'No data in ClientOverview'), exc_info=True)
            raise UserIdentityError  # E02
        client_package = client_package[0]
        self.client_id = client_package.client_id
        self.client_secret = client_package.client_secret
        self.des_key = client_package.des_key

    def encrypt(self, data):
        json_data = json.dumps(data)
        des_data = Cryption.aes_base64_encrypt(json_data, self.des_key)
        return des_data

    def _decrypt(self):
        try:
            json_data = Cryption.aes_base64_decrypt(self.origin_data, self.des_key)
            message = json.loads(json_data)
        except Exception as e:
            logger.error('Response from the function of `judge._decrypt`, error_msg=%s, rel_err_msg=%s'
                         % (EncryptError.message, e.message), exc_info=True)
            raise EncryptError  # E03

        self.apply_id = message.get('apply_id', None)

        if not self.apply_id:
            logger.error('Response from the function of `judge._decrypt`, error_msg=%s, rel_err_msg=%s'
                         % (GetApplyIdError.message, "Missing apply_id in the post_data"), exc_info=True)
            raise GetApplyIdError  # E04

        self.target_features = message.get('res_keys', None)

        if not self.target_features:
            logger.error('Response from the function of `judge._decrypt`, error_msg=%s, rel_err_msg=%s'
                         % (GetResKeysError.message, "Missing res_keys in the post_data"), exc_info=True)
            raise GetResKeysError  # E05

        apply_base = ApplyContext(self.apply_id)
        self.arguments = apply_base.load()

        if not self.arguments:
            logger.error('Response from the function of `judge._decrypt`, error_msg=%s, rel_err_msg=%s'
                         % (GetArgumentsError.message, "Missing arguments in the post_data"), exc_info=True)
            raise GetArgumentsError  # E06

    def _args_useful_check(self):
        """
        need sql which mapping the target features and arguments
        :return:
        """
        arg_msg_list = FeatureFieldRel.objects.filter(
            feature_name__in=self.target_features,
            is_delete=False,
        )
        for arg_msg in arg_msg_list:
            if arg_msg.raw_field_name in self.arguments.keys():
                if self.ret_msg and (arg_msg.feature_name == (self.ret_msg[-1])['target_field_name']):
                    sub_msg = self.ret_msg[-1]

                    if arg_msg.feature_name == sub_msg['target_field_name']:
                        sub_msg['arguments'].update({
                            arg_msg.raw_field_name: self.arguments[arg_msg.raw_field_name],
                        })
                        self.ret_msg[-1] = sub_msg
                else:
                    temp_msg = {
                        'data_identity': arg_msg.data_identity,
                        'target_field_name': arg_msg.feature_name,
                        'arguments': {
                            arg_msg.raw_field_name: self.arguments[arg_msg.raw_field_name],
                        }
                    }
                    self.ret_msg.append(temp_msg)
            else:
                logger.error('Response from the function of `judge._args_useful_check`, error_msg=%s, rel_err_msg=%s'
                             % (ArgumentsAvailableError.message, "Arguments are not enough to get all res_keys"),
                             exc_info=True)
                raise ArgumentsAvailableError  # E07

    def work_stream(self):
        self._check_identity()
        self._decrypt()
        self._args_useful_check()
        return self._check_sum()
