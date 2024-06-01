import json
import logging

from django.http import JsonResponse

logger = logging.getLogger('log')
from wxcloudrun.utils.SQL.DBUtils import DBUtils


def test1(request):
    print(request.headers)
    logger.info(request.headers)

    rsp = JsonResponse({'code': 0, 'errorMsg': '😁'}, json_dumps_params={'ensure_ascii': False})


    return rsp
