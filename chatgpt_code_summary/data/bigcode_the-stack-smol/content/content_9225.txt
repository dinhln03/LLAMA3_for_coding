
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

from cajas.movement.models.movement_withdraw import MovementWithdraw
from cajas.office.models.officeCountry import OfficeCountry
from cajas.users.models.partner import Partner
from cajas.webclient.views.utils import get_president_user

president = get_president_user()


class MovementWithdrawRequireList(LoginRequiredMixin, TemplateView):
    """
    """

    login_url = '/accounts/login/'
    redirect_field_name = 'redirect_to'
    template_name = 'webclient/movement_withdraw_require_list.html'

    def get_context_data(self, **kwargs):
        context = super(MovementWithdrawRequireList, self).get_context_data(**kwargs)
        movements = MovementWithdraw.objects.all()
        context['movements'] = movements
        context['all_offices'] = OfficeCountry.objects.all().order_by('office')
        context['partners_offices'] = Partner.objects.all().exclude(user=president)
        return context
