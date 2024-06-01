"""
Views for PubSite app.
"""
from django.conf import settings
from django.contrib.auth.views import (
    PasswordResetView,
    PasswordResetDoneView,
    PasswordResetConfirmView,
    PasswordResetCompleteView,
)
from django.shortcuts import render
import requests
import logging

logger = logging.getLogger(__name__)


def _get_context(page_name):
    return {
        "pages": settings.PUBLIC_PAGES,
        "current_page_name": page_name,
    }


# Regular index
# def index(request):
#     """
#     View for the static index page
#     """
#     return render(request, 'public/home.html', _get_context('Home'))


def index(request):
    """
    View for the static index page
    """
    return render(request, "public/home.html", _get_context("Home"))


def about(request):
    """
    View for the static chapter history page.
    """
    return render(request, "public/about.html", _get_context("About"))


def activities(request):
    """
    View for the static chapter service page.
    """
    return render(
        request,
        "public/activities.html",
        _get_context("Service & Activities"),
    )


def rush(request):
    """
    View for the static chapter service page.
    """
    return render(
        request,
        "public/rush.html",
        _get_context("Rush"),
    )


def campaign(request):
    """
    View for the campaign service page.
    """

    # Overrride requests Session authentication handling
    class NoRebuildAuthSession(requests.Session):
        def rebuild_auth(self, prepared_request, response):
            """
            No code here means requests will always preserve the Authorization
            header when redirected.
            Be careful not to leak your credentials to untrusted hosts!
            """

    url = "https://api.givebutter.com/v1/transactions/"
    headers = {"Authorization": f"Bearer {settings.GIVEBUTTER_API_KEY}"}
    response = None
    # Create custom requests session
    session = NoRebuildAuthSession()

    # Make GET request to server, timeout in seconds
    try:
        r = session.get(url, headers=headers, timeout=0.75)
        if r.status_code == 200:
            response = r.json()
        else:
            logger.error(f"ERROR in request: {r.status_code}")
    except requests.exceptions.Timeout:
        logger.warning("Connection to GiveButter API Timed out")
    except requests.ConnectionError:
        logger.warning("Connection to GiveButter API could not be resolved")
    except requests.exceptions.RequestException:
        logger.error(
            "An unknown issue occurred while trying to retrieve GiveButter Donor List"
        )

    # Grab context object to use later
    ctx = _get_context("Campaign")

    # Check for successful response, if so - filter, sort, and format data
    if response and "data" in response:
        response = response["data"]  # Pull data from GET response object
        logger.debug(f"GiveButter API Response: {response}")

        # Filter by only successful transactions, then sort by amount descending
        successful_txs = [tx for tx in response if tx["status"] == "succeeded"]
        sorted_txs = sorted(successful_txs, key=lambda tx: tx["amount"], reverse=True)

        # Clean data to a list of dictionaries & remove unnecessary data
        transactions = [
            {
                "name": tx["giving_space"]["name"],
                "amount": tx["giving_space"]["amount"],
                "message": tx["giving_space"]["message"],
            }
            for tx in sorted_txs[:20]
        ]

        # Attach transaction dictionary & length to context object
        ctx["transactions"] = transactions
        ctx["num_txs"] = len(successful_txs)

    return render(
        request,
        "public/campaign.html",
        ctx,
    )


def permission_denied(request):
    """
    View for 403 (Permission Denied) error.
    """
    return render(
        request,
        "common/403.html",
        _get_context("Permission Denied"),
    )


def handler404(request, exception):
    """ """
    return render(request, "common/404.html", _get_context("Page Not Found"))


class ResetPassword(PasswordResetView):
    template_name = "password_reset/password_reset_form.html"


class ResetPasswordDone(PasswordResetDoneView):
    template_name = "password_reset/password_reset_done.html"


class ResetPasswordConfirm(PasswordResetConfirmView):
    template_name = "password_reset/password_reset_confirm.html"


class ResetPasswordComplete(PasswordResetCompleteView):
    template_name = "password_reset/password_reset_complete.html"
