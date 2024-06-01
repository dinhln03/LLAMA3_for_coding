import mock
import json
from collections import OrderedDict
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from apps.common.tests import GetResponseMixin
from apps.issue.models import Issue, IssueStatus, IssueExtValue
from apps.user_group.models import UserGroupType
from gated_launch_backend.settings_test import JIRA_API_URL, JIRA_ZC_USER


class BusinessModulesRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/issue/fixtures/tests/business_modules.json"
    ]

    def test_list_business_modules(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='normal_user'))
        url = reverse('businessmodules-list')
        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_business_status_code(response), status.HTTP_200_OK)
        self.assertEqual(response.data,
                         OrderedDict([('status', 200), ('msg', '成功'),
                                      ('data', OrderedDict([('total', 2), ('next', None), ('previous', None),
                                                            ('results',
                                                             [OrderedDict([('id', 2), ('name', 'parking car'),
                                                                           ('level', 1), ('parent', 'parking'),
                                                                           ('parentId', 1), ('disabled', True)]),
                                                              OrderedDict([('id', 1), ('name', 'parking'),
                                                                           ('level', 0), ('parent', None),
                                                                           ('parentId', None),
                                                                           ('disabled', False)])])]))]))


class PhoneBrandsRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/issue/fixtures/tests/phone_brands.json"
    ]

    def test_list_business_modules(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='normal_user'))
        url = reverse('phonebrands-list')
        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_business_status_code(response), status.HTTP_200_OK)
        self.assertEqual(response.data,
                         OrderedDict([('status', 200), ('msg', '成功'),
                                      ('data', OrderedDict([('total', 1), ('next', None), ('previous', None),
                                                            ('results', [OrderedDict([('id', 1),
                                                                                      ('name', 'Huawei P8')])])]))]))


class RegionsRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/issue/fixtures/tests/regions.json"
    ]

    def test_list_business_modules(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='normal_user'))
        url = reverse('regions-list')
        response = self.client.get(url, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_business_status_code(response), status.HTTP_200_OK)
        self.assertEqual(response.data,
                         OrderedDict([('status', 200), ('msg', '成功'),
                                      ('data', OrderedDict([('total', 1), ('next', None), ('previous', None),
                                                            ('results', [OrderedDict([('id', 1),
                                                                                      ('name', 'North')])])]))]))


class IssuesRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/common/fixtures/tests/images.json",
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/user_group/fixtures/tests/user_groups.json",
        "apps/app/fixtures/tests/app_types.json",
        "apps/app/fixtures/tests/apps.json",
        "apps/app/fixtures/tests/app_components.json",
        "apps/task_manager/fixtures/tests/task_status.json",
        "apps/task_manager/fixtures/tests/info_api_test_graytask.json",
        "apps/task_manager/fixtures/tests/info_api_test_snapshotinnerstrategy.json",
        "apps/issue/fixtures/tests/report_sources.json",
        "apps/issue/fixtures/tests/issues.json",
        "apps/usage/fixtures/tests/usage_eventtype.json",
        "apps/usage/fixtures/tests/usage_eventtracking.json",
        "apps/usage/fixtures/tests/usage_property.json"
    ]

    def test_filter_issues_by_contain_creator(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))

        response = self.client.get(reverse('issues-list'), {'creator': 'normal_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'creator': 'normal_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'creator': 'admin_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'creator': 'admin_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'creator': 'app_owner_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'creator': 'app_owner_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_report_source(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'reportSource': 'weixin', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'reportSource': '四大区运营', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'reportSource': 'no_source', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_jira_id(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'jiraId': 'CC-157', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'jiraId': 'AAABBB', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'jiraId': 'AA-170', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_department(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'department': '网科集团', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '网科', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '质量管理部', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '质量', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '地产集团', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '不存在部门', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'department': '地产集团', 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '地产', 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '工程部', 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '工', 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'department': '不存在部门', 'appId': 2})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_priority(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'priority': '紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'priority': '一般', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'priority': '不紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 2)

    def test_filter_issues_by_status_order(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='admin_user'))

        # set app_owner_user as owner of app 6
        url = reverse('usergroups-list')
        data = {'type': UserGroupType.OWNER, 'appId': 6}
        response = self.client.get(url, data, format='json')

        group_id = response.data['data']['results'][0]['id']

        url = reverse('usergroupmems-list', kwargs={'group_id': group_id})
        data = {'account': 'app_owner_user'}
        self.client.post(url, data, format='json')

        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'statusNameOrder': '关闭,处理中,待处理,挂起,验证', 'appId': 6})

        expect_order = ['关闭', '处理中', '待处理', '挂起', '验证']
        # remove duplication
        real_order = OrderedDict.fromkeys([item['statusName'] for item in response.data['data']['results']]).keys()
        self.assertEqual(expect_order, list(real_order))

        response = self.client.get(reverse('issues-list'), {'statusNameOrder': '挂起,验证,关闭,处理中,待处理', 'appId': 6})

        expect_order = ['挂起', '验证', '关闭', '处理中', '待处理']
        real_order = OrderedDict.fromkeys([item['statusName'] for item in response.data['data']['results']]).keys()
        self.assertEqual(expect_order, list(real_order))

    def test_filter_issues_by_score_and_createdTime_startDate_endDate(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'score': 5, 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'score': 4, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'score': 5, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'score': 4, 'appId': 2, 'createdTime': '2017-07-01'})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'score': 4, 'appId': 2, 'createdTime': '2017-06-29'})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'score': 5, 'appId': 1, 'createdTime': '2017-07-01'})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'appId': 6, 'startDate': '2017-06-29', 'endDate': '2017-10-01'})
        self.assertEqual(self._get_response_total(response), 7)

        response = self.client.get(reverse('issues-list'), {'appId': 6, 'startDate': '2017-06-29', 'endDate': '2017-08-01'})
        self.assertEqual(self._get_response_total(response), 5)

    def test_filter_issues_by_multiple_score_value(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-list')

        # appId 1
        response = self.client.get(reverse('issues-list'), {'score': 5, 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'score': 4, 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        url_query = url + "?score=4&score=5&appId=1"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 2)

        url_query = url + "?score=4&score=5&appId=1&score=300000000"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 2)

        # appId 2
        response = self.client.get(reverse('issues-list'), {'score': 4, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'score': 5, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 0)

        url_query = url + "?score=4&score=5&appId=2&score=300000000"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 1)

    def test_create_issues_with_priority(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['priority'], '紧急')

        # no priority field
        response = self.client.post(reverse('issues-list'),
                                    {'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['priority'], '一般')

        # check result in backend
        response = self.client.get(reverse('issues-list'), {'priority': '紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issues-list'), {'priority': '一般', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

    def test_create_issues_with_report_source(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['score'], '0')

        # from weiXin: with reportSource and score field and reportSource field equal '四大区运营'
        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb', 'reportSource': '四大区运营',
                                     'score': '非常严重'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['score'], '5')

        # from weiXin: with reportSource field and no score filed and reportSource field equal '四大区运营'
        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb', 'reportSource': '四大区运营'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['score'], '4')

    def test_create_issues_with_updated_after(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'),
                                   {'appId': 1, 'updatedAfter': '1987-01-01 10:13:20'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_response_total(response), 2)

        response = self.client.get(reverse('issues-list'),
                                   {'appId': 1, 'updatedAfter': '2030-01-01 10:13:20'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_response_total(response), 0)

        date_time = '2017-06-29 20:25:00'
        response = self.client.get(reverse('issues-list'),
                                   {'appId': 1, 'updatedAfter': date_time})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_response_total(response), 0)

        # update 1 issue
        issue = Issue.objects.get(pk=1)
        issue.save()

        # filter with same updated time again
        response = self.client.get(reverse('issues-list'),
                                   {'appId': 1, 'updatedAfter': date_time})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_response_total(response), 1)

    def test_update_issues_with_priority(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 1})
        response = self.client.patch(url, {'priority': '紧急'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['priority'], '紧急')

        # check result in backend
        response = self.client.get(reverse('issues-list'), {'priority': '紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)
        response = self.client.get(reverse('issues-list'), {'priority': '不紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

    def test_update_issues_operator_no_jira(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 2})
        response = self.client.patch(url, {'operator': 'normal_user'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['operator'], 'manong')

    def test_update_issues_operator_exist_jira(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 1})
        response = self.client.patch(url, {'operator': 'normal_user'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['operator'], 'normal_user')

    def test_issue_stats_creator(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'creatorId': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['totalIssues'], 7)
        self.assertEqual(response.data['data']['results']['statusStats']['closed'], 2)

    def test_issue_stats_report_source(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'reportSource': '四大区运营'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['totalIssues'], 9)
        self.assertEqual(response.data['data']['results']['statusStats']['closed'], 2)

    def test_issue_stats_report_source_and_creator(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'reportSource': '四大区运营', 'creatorId': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['totalIssues'], 6)
        self.assertEqual(response.data['data']['results']['statusStats']['closed'], 2)

    def test_issue_stats_report_source_and_app(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'reportSource': '四大区运营', 'appId': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['totalIssues'], 1)
        self.assertEqual(response.data['data']['results']['statusStats']['closed'], 1)

    def test_issue_stats_report_source_and_task(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'reportSource': '四大区运营', 'taskId': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['totalIssues'], 8)
        self.assertEqual(response.data['data']['results']['statusStats']['closed'], 2)

    def test_issue_stats_test_filter_start_end_time(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        data = {'startTime': '2017-01-01'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 10)

        data = {'startTime': '2017-10-01', 'appId': 6}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 2)

        data = {'startTime': '2017-10-01', 'appId': 6, 'endTime': '2017-10-10'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 2)

        data = {'startTime': '2017-09-01', 'appId': 6, 'endTime': '2017-09-30'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 0)

        data = {'startTime': '2017-11-01', 'appId': 6, 'endTime': '2017-11-30'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 0)

    def test_issue_stats_test_valid_issues(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')
        response = self.client.get(url, format='json')
        self.assertEqual(response.data['data']['validIssues'], 4)

        data = {'appId': 6}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['validIssues'], 1)

        data = {'endTime': '2017-09-30'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['validIssues'], 3)

    def test_issue_stats_test_filter_issue_from(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issuestats')

        data = {'issueFrom': 'local'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 1)

        data = {'issueFrom': 'remote'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 1)

        data = {'issueFrom': 'fake_one'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 0)

        data = {'issueFrom': 'local', 'appId': 6}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 1)

        data = {'issueFrom': 'remote', 'appId': 6}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 1)

        data = {'issueFrom': 'local', 'appId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 0)

        data = {'issueFrom': 'remote', 'appId': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['totalIssues'], 0)

    def test_create_issues_with_extended_fields(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['extFields'], {'手机型号': '华为', '广场': '通州万达'})

    def test_can_not_create_issues_with_undefined_extended_fields(self):
        # 不能传入未定义的字段
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'场地': '通州万达', '手机型号': '华为'}},
                                    format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        OrderedDict([('status', 400),
                     ('msg', 'Not found:  IssueExtField matching query does not exist.')])

    def test_can_not_create_issues_without_must_have_extended_fields(self):
        # 必须的字段一定要有
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': False, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达'}},
                                    format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        OrderedDict([('status', 400), ('msg', "缺少以下必须扩展字段: {'手机型号'}")])

    def test_update_issues_with_extended_fields(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        issue_id = response.data['data']['id']
        url = reverse('issues-detail', kwargs={'pk': issue_id})
        response = self.client.patch(url, {'extFields': {'手机型号': '苹果'}},
                                     format='json')
        # 会全量更新扩展字段
        self.assertEqual(response.data['data']['extFields'], {'手机型号': '苹果'})

    def test_get_issue_extended_field_value_from_model_obj(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        issue_id = response.data['data']['id']

        issue_obj = Issue.objects.get(id=issue_id)
        self.assertEqual('通州万达', issue_obj.get_ext_field_value('广场'))
        self.assertEqual('华为', issue_obj.get_ext_field_value('手机型号'))

    def test_set_issue_extended_field_value_from_model_obj(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        issue_id = response.data['data']['id']

        issue_obj = Issue.objects.get(id=issue_id)
        self.assertTrue(issue_obj.set_ext_field_value('广场', '瞎写的广场'))
        self.assertEqual('瞎写的广场', issue_obj.get_ext_field_value('广场'))

        # 不影响其他字段
        self.assertEqual('华为', issue_obj.get_ext_field_value('手机型号'))

        self.assertTrue(issue_obj.set_ext_field_value('手机型号', '瞎写的手机型号'))
        self.assertEqual('瞎写的手机型号', issue_obj.get_ext_field_value('手机型号'))

        self.assertFalse(issue_obj.set_ext_field_value('瞎写的字段', 'aaa'))

    def test_delete_issue_will_delete_extended_fields(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场'})
        self.client.post(url, {'name': '手机型号', 'isOptional': True, 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        issue_id = response.data['data']['id']
        url = reverse('issues-detail', kwargs={'pk': issue_id})
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(self._get_business_status_code(response), status.HTTP_200_OK)

        self.assertEqual(0, IssueExtValue.objects.filter(issue_id=issue_id).count())

    def test_update_issues_will_check_extended_fields(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场', 'isOptional': False})
        self.client.post(url, {'name': '手机型号', 'default': 'iPhone', 'type': 'string'})

        response = self.client.post(reverse('issues-list'),
                                    {'priority': '紧急', 'appId': 1, 'taskId': 1,
                                     'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                                     'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                                    format='json')
        issue_id = response.data['data']['id']
        url = reverse('issues-detail', kwargs={'pk': issue_id})
        response = self.client.patch(url, {'extFields': {'手机型号': '苹果'}},
                                     format='json')
        self.assertEqual(response.data['status'], status.HTTP_400_BAD_REQUEST)

    def test_filter_issues_with_extended_fields(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueextfields-list', kwargs={'task_id': 1})
        self.client.post(url, {'name': '广场', 'isOptional': False})
        self.client.post(url, {'name': '手机型号', 'default': 'iPhone', 'type': 'string'})

        self.client.post(reverse('issues-list'),
                         {'priority': '紧急', 'appId': 1, 'taskId': 1,
                          'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                          'extFields': {'广场': '通州万达', '手机型号': '华为'}},
                         format='json')
        self.client.post(reverse('issues-list'),
                         {'priority': '紧急', 'appId': 1, 'taskId': 1,
                          'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                          'extFields': {'广场': '通州万达', '手机型号': '苹果'}},
                         format='json')
        self.client.post(reverse('issues-list'),
                         {'priority': '紧急', 'appId': 1, 'taskId': 1,
                          'statusId': 1, 'title': 'aaaa', 'detail': 'bbbb',
                          'extFields': {'广场': '大望路万达', '手机型号': '华为'}},
                         format='json')

        url = reverse('issues-list')
        data = {'广场': '大望路万达', 'appId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 1)

        data = {'广场': '大望路万达', 'appId': 6}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 0)

        data = {'广场': '通州万达', 'appId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

        data = {'手机型号': '华为', 'appId': 1, 'taskId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

        data = {'手机型号': '华为', '广场': '通州万达', 'appId': 1, 'taskId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 1)

        data = {'手机型号': '华为', '广场': '大望路万达'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 1)

        data = {'手机型号': '苹果', '广场': '大望路万达'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 0)

        data = {'广场': '大望路万达'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 1)

        data = {'手机型号': '华为'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

        data = {'广场': '通州万达'}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

        # check in pagination condition
        data = {'广场': '通州万达', 'pageSize': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

        data = {'广场': '通州万达', 'pageSize': 1, 'page': 2}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['total'], 2)

    def test_issue_component(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 1})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['componentName'], '技术支持')

        url = reverse('issues-detail', kwargs={'pk': 3})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['componentName'], '飞凡众测')

    def test_issue_operator_no_jira_link(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 8})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['jiraId'], '')
        self.assertEqual(response.data['data']['operator'], 'manong')

    def test_issue_operator_exist_jira_link(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issues-detail', kwargs={'pk': 1})
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['jiraId'], 'CC-157')
        self.assertEqual(response.data['data']['operator'], 'mingong')


def mocked_zc_set_jira_status(*args, **kwargs):
    return '待处理', ['status changed!']


def mocked_jira_issue_is_avaliable(*args, **kwargs):
    return True


# This method will be used by the mock to replace requests.post
def mocked_requests_post(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code
            self.data = json.dumps(self.json_data)

        def json(self):
            return self.json_data

    if args[0] == JIRA_API_URL:
        return MockResponse({'data': {'status': '待处理', 'jiraId': 'AA-157', 'operator': 'dalingdao'},
                             'status': 200}, 200)

    return MockResponse(None, 404)


class IssuesJiraRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/common/fixtures/tests/images.json",
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/user_group/fixtures/tests/user_groups.json",
        "apps/app/fixtures/tests/app_types.json",
        "apps/app/fixtures/tests/apps.json",
        "apps/task_manager/fixtures/tests/task_status.json",
        "apps/task_manager/fixtures/tests/info_api_test_graytask.json",
        "apps/task_manager/fixtures/tests/info_api_test_snapshotinnerstrategy.json",
        "apps/issue/fixtures/tests/report_sources.json",
        "apps/issue/fixtures/tests/issues.json",
    ]

    def setUp(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))

    @mock.patch('apps.issue.views.zc_set_jira_status', side_effect=mocked_zc_set_jira_status)
    @mock.patch('apps.issue.views.jira_issue_is_avaliable', side_effect=mocked_jira_issue_is_avaliable)
    def test_update_jira_comment_with_empty_jira_info(self, mock_obj_1, mock_obj_2):
        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'fail', 'comment': 'first comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "first comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'fail', 'comment': 'second comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "second comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

    @mock.patch('apps.issue.views.zc_set_jira_status', side_effect=mocked_zc_set_jira_status)
    def test_update_jira_comment_with_no_jira(self, mock_obj):
        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 2}),
                                    {'conclusion': '验证不通过'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['status'], "处理中")
        issue_with_pk_2 = Issue.objects.get(pk=2)
        self.assertEqual(issue_with_pk_2.status.name, "处理中")

        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 2}),
                                    {'conclusion': '验证通过'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['status'], "关闭")
        issue_with_pk_2 = Issue.objects.get(pk=2)
        self.assertEqual(issue_with_pk_2.status.name, "关闭")

    @mock.patch('apps.issue.views.zc_set_jira_status', side_effect=mocked_zc_set_jira_status)
    @mock.patch('apps.issue.views.jira_issue_is_avaliable', side_effect=mocked_jira_issue_is_avaliable)
    def test_update_jira_comment_with_jira_info_and_no_comments(self, mock_obj_1, mock_obj_2):
        issue_with_pk_1 = Issue.objects.get(pk=1)
        issue_with_pk_1.other = """{"phoneBrand": "华为 p8", "area": "四大区", "业务模块": "不知道写啥"}"""

        issue_with_pk_1.save()
        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'fail', 'comment': 'first comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "first comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'pass', 'comment': 'second comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "second comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

    @mock.patch('apps.issue.views.zc_set_jira_status', side_effect=mocked_zc_set_jira_status)
    @mock.patch('apps.issue.views.jira_issue_is_avaliable', side_effect=mocked_jira_issue_is_avaliable)
    def test_update_jira_comment_with_jira_info_and_comments(self, mock_obj_1, mock_obj_2):
        issue_with_pk_1 = Issue.objects.get(pk=1)
        issue_with_pk_1.other = """{"phoneBrand": "华为 p8", "area": "四大区", "业务模块": "不知道写啥",
        "comments": [{"wanxin": "app_owner_user", "email": "app_owner_user@test.com",
                    "name": "", "info": "presetting comment", "startTime": "", "endTime": ""}]}"""

        issue_with_pk_1.save()
        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'fail', 'comment': 'first comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "first comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'pass', 'comment': 'second comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "second comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")

    @mock.patch('apps.issue.views.zc_set_jira_status', side_effect=mocked_zc_set_jira_status)
    @mock.patch('apps.issue.views.jira_issue_is_avaliable', side_effect=mocked_jira_issue_is_avaliable)
    def test_update_jira_status(self, mock_obj_1, mock_obj_2):
        issue_with_pk_1 = Issue.objects.get(pk=1)
        issue_with_pk_1.status = IssueStatus.objects.get(name='验证')

        issue_with_pk_1.save()
        response = self.client.post(reverse('issues-jiracomment', kwargs={'pk': 1}),
                                    {'conclusion': 'fail', 'comment': 'first comment'})
        self.assertEqual(response.data['status'], 200)
        self.assertEqual(response.data['data']['comments'][0]['info'], "first comment")
        self.assertEqual(response.data['data']['comments'][0]['wanxin'], "app_owner_user")
        self.assertEqual(response.data['data']['changeLog'][0], "status changed!")
        issue_with_pk_1.refresh_from_db()
        self.assertEqual(issue_with_pk_1.status.name, '待处理')

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_create_jira(self, mock_post):
        url = reverse('issuetojira')
        data = {'issueId': 1}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['jiraId'], 'CC-157')

        data = {'issueId': 8}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.data['data']['jiraId'], 'AA-157')

    def test_jira_to_zc_jira_not_exist(self):
        url = reverse('jiratoissue')
        data = {
            "issue": {
                "key": "CC-15"
            },
            "user": {
                "name": "zhongce"
            }
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['issueId'], None)
        self.assertEqual(response.data['data']['jiraId'], 'CC-15')

    def test_jira_to_zc_user_is_zhongce(self):
        url = reverse('jiratoissue')
        data = {
            "issue": {
                "key": "CC-157"
            },
            "user": {
                "name": JIRA_ZC_USER
            }
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['issueId'], 1)
        self.assertEqual(response.data['data']['jiraId'], 'CC-157')

    def test_jira_to_zc_user_is_not_zhongce(self):
        url = reverse('jiratoissue')
        data = {
            "issue": {
                "key": "CC-157"
            },
            "user": {
                "name": "zhaochunyan7"
            }
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['issueId'], 1)
        self.assertEqual(response.data['data']['jiraId'], 'CC-157')

    def test_generate_change_log_jira_not_exist_update_priority(self):
        url = reverse('issues-detail', kwargs={'pk': 8})
        response = self.client.patch(url, {'priority': '紧急'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        zc_change_logs = json.loads(response.data['data']['zcChangeLogs'])
        zc_change_logs[0].pop('created')
        self.assertEqual(zc_change_logs, [{'wanxin': 'app_owner_user',
                                           'items': [{'field': 'priority', 'toString': '紧急', 'fromString': '不紧急'}],
                                           'author': ''}])

    def test_generate_change_log_jira_not_exist_update_images(self):
        url = reverse('issues-detail', kwargs={'pk': 8})
        response = self.client.patch(url, {'images': ['aabbceadfdfdfdfdfdf']})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        zc_change_logs = json.loads(response.data['data']['zcChangeLogs'])
        zc_change_logs[0].pop('created')
        self.assertEqual(zc_change_logs, [{'wanxin': 'app_owner_user',
                                           'items': [{'field': 'images',
                                                      'toString': "['aabbceadfdfdfdfdfdf']",
                                                      'fromString': '[]'}],
                                           'author': ''}])

    @mock.patch('requests.post', side_effect=mocked_requests_post)
    def test_generate_change_log_create_jira(self, mock_post):
        url = reverse('issuetojira')
        data = {'issueId': 8}
        response = self.client.get(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['jiraId'], 'AA-157')

        issue = Issue.objects.get(pk=8)
        zc_change_logs = json.loads(issue.zc_change_logs)
        zc_change_logs[0].pop('created')
        self.assertEqual(zc_change_logs, [{'author': '', 'wanxin': 'app_owner_user',
                                           'items': [{'fromString': '', 'toString': 'AA-157', 'field': 'jira link'}]}])


class IssuesLiteRESTTestCase(APITestCase, GetResponseMixin):
    fixtures = [
        "apps/common/fixtures/tests/images.json",
        "apps/auth/fixtures/tests/departments.json",
        "apps/auth/fixtures/tests/users.json",
        "apps/user_group/fixtures/tests/user_groups.json",
        "apps/app/fixtures/tests/app_types.json",
        "apps/app/fixtures/tests/apps.json",
        "apps/task_manager/fixtures/tests/task_status.json",
        "apps/task_manager/fixtures/tests/info_api_test_graytask.json",
        "apps/task_manager/fixtures/tests/info_api_test_snapshotinnerstrategy.json",
        "apps/issue/fixtures/tests/report_sources.json",
        "apps/issue/fixtures/tests/issues.json",
    ]

    def test_filter_issues_by_contain_creator(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))

        response = self.client.get(reverse('issueslite-list'), {'creator': 'normal_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'creator': 'normal_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'creator': 'admin_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'creator': 'admin_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'creator': 'app_owner_user', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issueslite-list'), {'creator': 'app_owner_', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_report_source(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issueslite-list'), {'reportSource': 'weixin', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'reportSource': '四大区运营', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'reportSource': 'no_source', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

    def test_filter_issues_by_priority(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issues-list'), {'priority': '紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'priority': '一般', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 0)

        response = self.client.get(reverse('issues-list'), {'priority': '不紧急', 'appId': 1})
        self.assertEqual(self._get_response_total(response), 2)

    def test_filter_issues_by_status_order(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='admin_user'))

        # set app_owner_user as owner of app 6
        url = reverse('usergroups-list')
        data = {'type': UserGroupType.OWNER, 'appId': 6}
        response = self.client.get(url, data, format='json')

        group_id = response.data['data']['results'][0]['id']

        url = reverse('usergroupmems-list', kwargs={'group_id': group_id})
        data = {'account': 'app_owner_user'}
        self.client.post(url, data, format='json')

        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issueslite-list'),
                                   {'statusNameOrder': '关闭,处理中,待处理,挂起,验证', 'appId': 6})

        expect_order = ['关闭', '处理中', '待处理', '挂起', '验证']
        # remove duplication
        real_order = OrderedDict.fromkeys([item['statusName'] for item in response.data['data']['results']]).keys()
        self.assertEqual(expect_order, list(real_order))

        response = self.client.get(reverse('issueslite-list'),
                                   {'statusNameOrder': '挂起,验证,关闭,处理中,待处理', 'appId': 6})

        expect_order = ['挂起', '验证', '关闭', '处理中', '待处理']
        real_order = OrderedDict.fromkeys([item['statusName'] for item in response.data['data']['results']]).keys()
        self.assertEqual(expect_order, list(real_order))

    def test_created_time_order_when_filter_issues_by_status_order_created_time(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='admin_user'))

        # set app_owner_user as owner of app 6
        url = reverse('usergroups-list')
        data = {'type': UserGroupType.OWNER, 'appId': 6}
        response = self.client.get(url, data, format='json')

        group_id = response.data['data']['results'][0]['id']

        url = reverse('usergroupmems-list', kwargs={'group_id': group_id})
        data = {'account': 'app_owner_user'}
        self.client.post(url, data, format='json')

        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        response = self.client.get(reverse('issueslite-list'),
                                   {'statusNameOrder': '关闭,处理中,待处理,挂起,验证', 'appId': 6})

        result = [(item['statusName'], item['createdAt']) for item in response.data['data']['results']]
        self.assertEqual(result, [('关闭', '2017-06-29T18:25:11.681308'),
                                  ('处理中', '2017-06-29T18:25:11.681308'),
                                  ('待处理', '2017-06-29T18:25:11.681308'), ('待处理', '2017-10-01T18:22:11.681308'),
                                  ('待处理', '2017-10-01T18:25:11.681308'),
                                  ('挂起', '2017-06-29T18:25:11.681308'),
                                  ('验证', '2017-06-29T18:25:11.681308')])

    def test_filter_issues_by_multiple_score_value(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))
        url = reverse('issueslite-list')

        # appId 1
        response = self.client.get(reverse('issueslite-list'), {'score': 5, 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'score': 4, 'appId': 1})
        self.assertEqual(self._get_response_total(response), 1)

        url_query = url + "?score=4&score=5&appId=1"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 2)

        url_query = url + "?score=4&score=5&appId=1&score=300000000"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 2)

        # appId 2
        response = self.client.get(reverse('issueslite-list'), {'score': 4, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 1)

        response = self.client.get(reverse('issueslite-list'), {'score': 5, 'appId': 2})
        self.assertEqual(self._get_response_total(response), 0)

        url_query = url + "?score=4&score=5&appId=2&score=300000000"
        response = self.client.get(url_query)
        self.assertEqual(self._get_response_total(response), 1)

    def test_issues_response(self):
        self.client.force_authenticate(user=get_user_model().objects.get(username='app_owner_user'))

        # appId 1
        response = self.client.get(reverse('issueslite-list'), {'score': 5, 'appId': 1})
        self.assertEqual(response.data,
                         OrderedDict([('status', 200),
                                      ('msg', '成功'),
                                      ('data', OrderedDict([('total', 1), ('next', None),
                                                            ('previous', None),
                                                            ('results',
                                                             [OrderedDict([('id', 1),
                                                                           ('jiraId', 'CC-157'),
                                                                           ('statusName', '待处理'), ('title', ''),
                                                                           ('createdAt', '2017-06-29T18:25:11.681308'),
                                                                           ('other', '{"phoneNumber":"15921372222","order":"12345678","phoneType":"P9","version":"0928gray","square":"通州万达","summary":"example全量数据","description":"example全量数据","occurrenceTime":"2017-09-01T09:01:00.000+0800","area":"ALL","phoneBrand":"华为","severity":"次要","businessType":"停车"}'),  # noqa
                                                                           ('score', 5), ('remindKSTFlag', False),
                                                                           ('remindPlatFlag', False)])])]))]))
