from api.views import BucketlistViewSet, ItemViewSet, UserViewSet
from django.conf.urls import url, include
from rest_framework.authtoken import views as authviews
from rest_framework_nested import routers

router = routers.SimpleRouter()
router.register(r'bucketlists', BucketlistViewSet)
router.register(r'users', UserViewSet)

bucketlists_router = routers.NestedSimpleRouter(router, r'bucketlists',
                                                lookup='bucketlist')
bucketlists_router.register(r'items', ItemViewSet)

urlpatterns = [
    url(r'^', include(router.urls)),
    url(r'^', include(bucketlists_router.urls)),
    url(r'^auth/login/', authviews.obtain_auth_token, name='api-login'),
]
