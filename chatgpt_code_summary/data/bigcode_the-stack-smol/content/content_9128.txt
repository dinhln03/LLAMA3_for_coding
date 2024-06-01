import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from talents.models import Agency, Talent
from orders.models import (
    AgencyProfit,
    AgencyProfitPercentage,
    Buyer,
    Charge,
    CreditCard,
    CustomTalentProfitPercentage,
    TalentProfit,
    DefaultTalentProfitPercentage,
    Order,
)
from request_shoutout.domain.models import Charge as DomainCharge
from shoutouts.models import ShoutoutVideo
from utils.telegram import TELEGRAM_BOT_API_URL
from wirecard.models import WirecardTransactionData

User = get_user_model()

FAKE_WIRECARD_ORDER_HASH = 'ORD-O5DLMAJZPTHV'
FAKE_WIRECARD_PAYMENT_HASH = 'PAY-HL7QRKFEQNHV'


def get_wirecard_mocked_abriged_response():
    wirecard_capture_payment_api_abriged_response = {
        'id': FAKE_WIRECARD_PAYMENT_HASH,
        'status': 'AUTHORIZED',
    }
    capture_payment_response = mock.Mock()
    capture_payment_response.status_code = 200
    capture_payment_response.json.return_value = wirecard_capture_payment_api_abriged_response
    return capture_payment_response


@override_settings(
    task_eager_propagates=True,
    task_always_eager=True,
    broker_url='memory://',
    backend='memory'
)
@mock.patch('wirecard.services.requests.post', return_value=get_wirecard_mocked_abriged_response())
class FulfillShoutoutRequestTest(APITestCase):

    def do_login(self, user, password):
        data = {
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'password': password,
        }
        response = self.client.post(reverse('accounts:signin'), data, format='json')
        token = response.data['access']
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')

    def setUp(self):
        self.maxDiff = None
        password = 'senha123'
        user = User(
            email='talent1@viggio.com.br',
            first_name='Nome',
            last_name='Sobrenome',
        )
        user.set_password(password)
        user.save()
        self.do_login(user, password)
        self.talent = Talent.objects.create(
            user=user,
            price=1000,
            phone_number=1,
            area_code=1,
            main_social_media='',
            social_media_username='',
            number_of_followers=1,
        )
        self.order = Order.objects.create(
            hash_id=uuid.uuid4(),
            talent_id=self.talent.id,
            video_is_for='someone_else',
            is_from='MJ',
            is_to='Peter',
            instruction="Go Get 'em, Tiger",
            email='mary.jane.watson@spiderman.com',
            is_public=True,
            expiration_datetime=datetime.now(timezone.utc) + timedelta(days=4),
        )
        charge = Charge.objects.create(
            order=self.order,
            amount_paid=1000,
            payment_date=datetime.now(timezone.utc) - timedelta(days=3),
            status=DomainCharge.PRE_AUTHORIZED,
        )
        CreditCard.objects.create(
            charge=charge,
            fullname='Peter Parker',
            birthdate='2019-12-31',
            tax_document='12346578910',
            credit_card_hash='<encrypted-credit-card-hash>',
        )
        Buyer.objects.create(
            charge=charge,
            fullname='Mary Jane Watson',
            birthdate='2019-12-31',
            tax_document='09876543210',
        )
        WirecardTransactionData.objects.create(
            order=self.order,
            wirecard_order_hash=FAKE_WIRECARD_ORDER_HASH,
            wirecard_payment_hash=FAKE_WIRECARD_PAYMENT_HASH,
        )
        DefaultTalentProfitPercentage.objects.create(value='0.75')
        self.request_data = {
            'talent_id': self.talent.id,
            'order_hash': self.order.hash_id,
            'order_video': SimpleUploadedFile("file.mp4", b"filecontentstring"),
        }
        self.agency = Agency.objects.create(name='Agency')
        AgencyProfitPercentage.objects.create(agency=self.agency, value='0.05')

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_fulfilling_a_shoutout_request_create_a_shoutout_video(self, mock1):
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(ShoutoutVideo.objects.count(), 1)
        shoutout = ShoutoutVideo.objects.first()
        expected_file_url = f'orders/talent-1/order-{shoutout.order.hash_id}/viggio-para-peter.mp4'
        self.assertEqual(shoutout.hash_id, response.data['shoutout_hash'])
        self.assertTrue(shoutout.file.url.endswith(expected_file_url))

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_fulfilling_a_shoutout_request_create_a_talent_profit(self, mock1):
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(TalentProfit.objects.count(), 1)
        talent_profit_qs = TalentProfit.objects.filter(
            talent=self.talent,
            order=self.order,
            shoutout_price=1000,
            profit_percentage=Decimal('0.75'),
            profit=Decimal('750.00'),
            paid=False
        )
        self.assertTrue(talent_profit_qs.exists())

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_fulfilling_a_shoutout_request_create_a_agency_profit_when_talent_is_managed(self, mock1):  # noqa: E501
        self.talent.agency = self.agency
        self.talent.save()

        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        self.assertEqual(AgencyProfit.objects.count(), 1)
        agency_profit_qs = AgencyProfit.objects.filter(
            agency=self.agency,
            order=self.order,
            shoutout_price=1000,
            profit_percentage=Decimal('0.05'),
            profit=Decimal('50.00'),
            paid=False
        )
        self.assertTrue(agency_profit_qs.exists())

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_fulfilling_a_shoutout_request_dont_create_a_agency_profit_when_talent_isnt_managed(self, mock1):  # noqa: E501
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(AgencyProfit.objects.count(), 0)

    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_after_upload_a_shoutout_transcode_process_is_triggered(self, mock1):
        with mock.patch('transcoder.tasks.transcode') as mocked_transcoder:
            response = self.client.post(
                reverse('request_shoutout:fulfill'),
                self.request_data,
                format='multipart'
            )
        self.assertEqual(ShoutoutVideo.objects.count(), 1)
        shoutout = ShoutoutVideo.objects.first()
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        mocked_transcoder.assert_called_once_with(shoutout, 'mp4')

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    def test_send_email_to_customer_after_transcode_process_ending(self, mock1):
        with mock.patch('post_office.mailgun.requests') as mocked_requests:
            response = self.client.post(
                reverse('request_shoutout:fulfill'),
                self.request_data,
                format='multipart'
            )
        shoutout = ShoutoutVideo.objects.first()
        expected_calls = [
            mock.call(
                auth=('api', os.environ['MAILGUN_API_KEY']),
                url=os.environ['MAILGUN_API_URL'],
                data={
                    'from': os.environ['CONTACT_EMAIL'],
                    'to': 'MJ <mary.jane.watson@spiderman.com>',
                    'subject': 'Seu viggio para Peter está pronto',
                    'template': 'notify-customer-that-his-viggio-is-ready',
                    'v:order_is_to': 'Peter',
                    'v:customer_name': 'MJ',
                    'v:talent_name': 'Nome Sobrenome',
                    'v:shoutout_absolute_url': f'{os.environ["SITE_URL"]}v/{shoutout.hash_id}'
                },
            ),
        ]
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(mocked_requests.post.mock_calls, expected_calls)

    @mock.patch('request_shoutout.adapters.db.orm.DjangoTalentProfit.persist', side_effect=Exception())
    def test_rollback_when_fulfilling_a_shoutout_request_fails(self, mock1, mock2):
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(
            response.data,
            {'error': 'It happened an issue when persisting shoutout video'},
        )
        self.assertEqual(TalentProfit.objects.count(), 0)
        self.assertEqual(ShoutoutVideo.objects.count(), 0)

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    def test_when_talent_profit_percentage_is_not_the_default(self, mock1):
        CustomTalentProfitPercentage.objects.create(talent=self.talent, value=Decimal('0.80'))
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(TalentProfit.objects.count(), 1)
        talent_profit_qs = TalentProfit.objects.filter(
            talent=self.talent,
            order=self.order,
            shoutout_price=1000,
            profit_percentage=Decimal('0.80'),
            profit=Decimal('800.00'),
            paid=False
        )
        self.assertTrue(talent_profit_qs.exists())

    def test_cant_fulfill_same_order_twice(self, mock1):
        ShoutoutVideo.objects.create(
            hash_id=uuid.uuid4(),
            order=self.order,
            talent=self.talent,
            file=SimpleUploadedFile("file.mp4", b"filecontentstring"),
        )
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {'error': 'Order already has a shoutout attached.'})

    def test_cant_fulfill_an_expired_order(self, mock1):
        self.order.expiration_datetime = datetime.now(timezone.utc) - timedelta(hours=1)
        self.order.save()
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {'error': "Can't fulfill an expired order."})

    def test_a_talent_cant_fulfill_an_order_requested_to_another_talent(self, mock1):
        user = User.objects.create(email='talent100@youtuber.com')
        talent = Talent.objects.create(
            user=user,
            price=10,
            phone_number=1,
            area_code=1,
            main_social_media='',
            social_media_username='',
            number_of_followers=1,
        )
        self.order.talent_id = talent.id
        self.order.save()
        response = self.client.post(
            reverse('request_shoutout:fulfill'),
            self.request_data,
            format='multipart'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data, {'error': 'Order belongs to another Talent.'})

    @mock.patch('transcoder.tasks.transcode', mock.Mock())
    @mock.patch('post_office.mailgun.requests', mock.Mock())
    @mock.patch('utils.telegram.requests.post')
    def test_when_capture_payment_fails_it_should_send_alert_message_to_staff(self, mock1, telegram_request_post):  # noqa: E501
        expected_call = mock.call(
            url=f'{TELEGRAM_BOT_API_URL}/sendMessage',
            data=json.dumps({
                'chat_id': os.environ['TELEGRAM_GROUP_ID'],
                'text': (
                    'OCORREU UM ERRO AO CAPTURAR UM PAGAMENTO. '
                    'Verifique o Sentry: '
                    'https://sentry.io/organizations/viggio-sandbox/issues/?project=1770932'
                )
            }),
            headers={'Content-Type': 'application/json'}
        )
        method_path = 'request_shoutout.adapters.db.orm.WirecardPaymentApi.capture_payment'
        with mock.patch(method_path, side_effect=Exception):
            response = self.client.post(
                reverse('request_shoutout:fulfill'),
                self.request_data,
                format='multipart'
            )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(telegram_request_post.mock_calls, [expected_call])
