from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
import requests

from . import exceptions


class Gateway(models.Model):
    label = models.CharField(max_length=255, verbose_name=_('Label'))
    api_key = models.CharField(max_length=255, verbose_name=_('API Key'))
    default_callback = models.CharField(max_length=255, null=True, blank=True, verbose_name=_('Redirect to'), help_text=_('Enter the path name for a view that will verify the transaction.'))

    class Meta:
        verbose_name = _('Gateway')
        verbose_name_plural = _('Gateways')

    submission_url = 'https://pay.ir/pg/send'
    verification_url = 'https://pay.ir/pg/verify'

    def _prepare_submission_payload(self, request, transaction, mobile, valid_card_number, callback):
        if callback is None:
            raise ValueError('You need to specify a path name as the callback for your transactions.')
        return {
            'api': self.api_key,
            'amount': transaction.amount,
            'redirect': request.build_absolute_uri(reverse(callback)),
            'mobile': mobile,
            'factorNumber': transaction.id,
            'description': transaction.description,
            'validCardNumber': valid_card_number
        }

    def submit(self, request, transaction, mobile: str = None, valid_card_number: str = None, callback: str = None):
        """Submits a transaction to Pay.ir.

        When called, the method submits the necessary information about the transaction to Pay.ir and returns a
        HttpResponseRedirect object that can redirect the user to the gateway, if nothing goes wrong. In case of an
        error, a GatewayError is raised, containing the error_code and error_message reported by Pay.ir.

        :param request: The WSGIRequest object passed to the view.
        :param transaction: A transaction object (or a similar class) that's already been saved to the database.
        :param mobile: (Optional) Phone number of the payer. If provided, payer's saved card numbers will be listed for them in the gateway.
        :param valid_card_number: (Optional) Specifies a single card number as the only one that can complete the transaction.
        :param callback: (Optional) Overrides the default callback of the gateway.
        """
        payload = self._prepare_submission_payload(request, transaction, mobile, valid_card_number, callback or self.default_callback)
        response = requests.post(self.submission_url, data=payload)
        data = response.json()
        if response:
            transaction.token = data['token']
            transaction.save()
            return redirect(f'https://pay.ir/pg/{transaction.token}')
        raise exceptions.GatewayError(error_code=data['errorCode'], error_message=data['errorMessage'])

    def create_and_submit(self, request, account, amount: int, mobile: str = None, valid_card_number: str = None, callback: str = None):
        """Creates a transaction object and submits the transaction to Pay.ir.

        When called, the method submits the necessary information about the transaction to Pay.ir and returns a
        HttpResponseRedirect object that can redirect the user to the gateway, if nothing goes wrong. In case of an
        error, a GatewayError is raised, containing the error_code and error_message reported by Pay.ir.

        :param request: The WSGIRequest object passed to the view.
        :param account: Payer's account object. The account will be assigned to the transaction through a ForeignKey.
        :param amount: The amount of the transaction in IRR. The amount has to be more than 1000.
        :param mobile: (Optional) Phone number of the payer. If provided, payer's saved card numbers will be listed for them in the gateway.
        :param valid_card_number: (Optional) Specifies a single card number as the only one that can complete the transaction.
        :param callback: (Optional) Overrides the default callback of the gateway.
        """
        transaction = Transaction(account=account, amount=amount)
        transaction.save()
        return self.submit(request, transaction, mobile, valid_card_number, callback)

    def verify(self, transaction):
        """Verifies the transaction with Pay.ir.

        When a transaction returns with status '1', it must be verified with Pay.ir. Otherwise, it will be returned to
        the payer's bank account in 30 minutes. The method returns the updated transaction object and a boolean value.
        The boolean value would be True if the `verified` flag of the transaction was switched to True. If the
        `verified` attribute of transaction object and the returned boolean value do not match, the user might be trying
        to confirm a payment for a second time.

        :param transaction: The transaction object corresponding to the specified token in request.GET.
        """
        payload = {'api': self.api_key, 'token': transaction.token}
        response = requests.post(self.verification_url, data=payload)
        data = response.json()
        if response:
            if not transaction.verified:
                transaction.gateway = self
                transaction.verified = True
                transaction.verified_at = timezone.now()
                transaction.save()
                return transaction, True
            else:
                return transaction, False
        raise exceptions.GatewayError(error_code=data['errorCode'], error_message=data['errorMessage'])

    def find_and_verify(self, token: str):
        """Finds a transaction with a matching token value and verifies it with Pay.ir.

        When a transaction returns with status '1', it must be verified with Pay.ir. Otherwise, it will be returned to
        the payer's bank account in 30 minutes. The method returns the updated transaction object and a boolean value.
        The boolean value would be True if the `verified` flag of the transaction was switched to True. If the
        `verified` attribute of transaction object and the returned boolean value do not match, the user might be trying
        to confirm a payment for a second time.

        :param token: The token of the transaction, which can be found in request.GET. The method will look for a
        transaction object with the same token and return it as the first argument.
        """
        transaction = Transaction.objects.get(token=token)
        return self.verify(transaction)

    def __str__(self):
        return self.label


class Transaction(models.Model):
    account = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name=_('Account'))
    created = models.DateTimeField(auto_now_add=True, auto_now=False, verbose_name=_('Created'))
    modified = models.DateTimeField(auto_now=True, verbose_name=_('Modified'))
    amount = models.IntegerField(verbose_name=_('Amount (IRR)'))
    description = models.CharField(max_length=255, null=True, blank=True, verbose_name=_('Description'))
    gateway = models.ForeignKey(to=Gateway, on_delete=models.SET_NULL, null=True, blank=True, verbose_name=_('Gateway'))
    token = models.TextField(null=True, blank=True, unique=True, verbose_name=_('Token'))
    verified = models.BooleanField(default=False, verbose_name=_('Verified'))
    verified_at = models.DateTimeField(null=True, blank=True, verbose_name=_('Verified At'))

    class Meta:
        ordering = ['-modified']
        verbose_name = _('Transaction')
        verbose_name_plural = _('Transactions')

    def __str__(self):
        return _('Transaction %(id)d') % {'id': self.id}
