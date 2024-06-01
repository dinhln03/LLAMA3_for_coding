
from django.contrib.auth.decorators import login_required
from django.utils.decorators        import method_decorator

from selectelhackaton.utils              import SimpleAuthMixinView

@method_decorator(login_required, name='get')
class MemberIndex(SimpleAuthMixinView):
    template_name = 'member/member-index.html'


