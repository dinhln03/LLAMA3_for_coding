from datetime import datetime, timedelta

import requests
from decouple import config
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from .models import Socio


class ModelTest(TestCase):
    def setUp(self):
        Socio(
            user=User.objects.create_user(
                username='00000000',
                password='000000'
            ),
            nome='João de Souza',
            apelido='João',
            whatsapp='(86) 9 9123-4567',
            cpf='068.008.773-79',
            rg='123456789',
            data_nascimento='2000-01-01',
            data_inicio=timezone.now(),
            data_fim=timezone.now() + timedelta(days=40),
            is_socio=True,
            stripe_customer_id='cus_00000000',).save()

    def test_notificar_email(self):
        socio = Socio.objects.create(
            user=User.objects.create_user(
                username='12345678',
                password='123456',
            ),
            nome='Fulano',
            stripe_customer_id='cus_123456789',
        )

        notificar = socio.notificar(metodo='email', mensagem='teste')

        self.assertEqual(notificar, 'Enviando email...')

    def test_datetime(self):
        current_period_end = datetime(
            2022, 6, 30, 23, 59, 59
        )
        if current_period_end - datetime.now() > timedelta(days=30):
            if datetime.now().month < 7:
                if current_period_end.month > 6:
                    current_period_end = datetime(
                        datetime.now().year, 6, 30, 23, 59, 59
                    )

    def test_adicionar_socio_cheers(self):
        socio: Socio = Socio.objects.get(user__username='00000000')

        if socio.data_fim - timezone.now().date() > timedelta(days=30) and socio.is_socio:
            url = 'https://cheersshop.com.br/socio/adicionar'
            obj = {
                "nome": socio.nome,
                "email": socio.email,
                "telefone": socio.whatsapp,
                "matricula": socio.matricula,
                "observacao": "",
                "cpf": socio.cpf,
                "data_fim_plano": socio.data_fim,
                "vendedor": "1874"
            }

            response = requests.post(url, data=obj, headers={
                'Authorization': f'Bearer {config("CHEERS_TOKEN")}'})

            self.assertEqual(response.status_code, 200)

    def test_adicionar_coupom_cheers(self):
        socio: Socio = Socio.objects.get(user__username='00000000')

        if socio.is_socio:
            url = 'https://cheersshop.com.br/codigo'
            obj = {
                "nome": socio.cpf,
                "uso": 1,
                "ativo": True,
                "desconto_reais": 70 if socio.is_atleta else 65,
                "maximo_usuario": "1",
                "quantidade": "1",
                "usuario": 192061,
                "vendedor": "1874",
            }

            response = requests.post(url, data=obj, headers={
                'Authorization': f'Bearer {config("CHEERS_TOKEN")}'})

            self.assertEqual(response.json()['status'], 'Success')
