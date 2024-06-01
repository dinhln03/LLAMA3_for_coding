from django.http import JsonResponse, HttpResponseRedirect
from rest_framework.decorators import api_view

from sdk.key_generation import generate_random_key
from sdk.storage import create_storage
from sdk.url import URL, ModelValidationError

storage = create_storage()


@api_view(['GET'])
def go_to(request, key, format=None):
    url = storage.get(key)
    if not url:
        return JsonResponse(status=404, data={
            'error': 'key not found'
        })

    return HttpResponseRedirect(redirect_to=url.address)


@api_view(['POST'])
def shorten(request, format=None):
    raw_url = request.data.get('url')
    if not raw_url:
        return JsonResponse(status=400, data={
            'error': 'missing url parameter'
        })

    try:
        url = URL.parse(raw_url)
    except ModelValidationError as e:
        return JsonResponse(status=400, data={
            'error': 'invalid URL',
            'details': e.message
        })

    key = _store_url_and_get_key(url)
    return JsonResponse(status=200, data={
        'key': key
    })


def _store_url_and_get_key(url):
    while True:
        key = generate_random_key()
        if storage.set(key, url):
            break

    return key
