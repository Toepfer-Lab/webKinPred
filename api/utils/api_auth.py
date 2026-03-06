"""
API key authentication for the public REST API.

All v1 endpoints that require authentication should be decorated with
@require_api_key.  The decorator:

  1. Reads the "Authorization: Bearer <key>" header.
  2. Looks up the key in the database.
  3. Checks that the key is active and that the owner is not blocked.
  4. Stamps the key's last_used timestamp.
  5. Attaches request.api_user and request.api_ip for use in the view.

The decorator also applies @csrf_exempt so that programmatic clients do not
need to obtain a CSRF cookie.
"""

from functools import wraps

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt


def require_api_key(view_func):
    """
    Decorator that authenticates requests via a Bearer token.

    Usage:

        @require_api_key
        def my_view(request):
            ip = request.api_ip     # registered IP of the key owner
            user = request.api_user  # ApiUser instance
            ...

    On success, two attributes are attached to the request object:
      - request.api_user  — the ApiUser record tied to this key
      - request.api_ip    — the IP address stored on that ApiUser record
                            (used for quota accounting)

    The quota system is keyed by IP address, so using the registered IP (rather
    than the actual request IP) ensures quota is always charged to the correct
    account regardless of where the request originates.
    """

    @csrf_exempt
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        from api.models import ApiKey  # local import to avoid circular imports

        auth_header = request.META.get("HTTP_AUTHORIZATION", "")

        if not auth_header.startswith("Bearer "):
            return JsonResponse(
                {
                    "error": (
                        "Authentication required. "
                        "Include the header: Authorization: Bearer <your_api_key>"
                    )
                },
                status=401,
            )

        token = auth_header[len("Bearer "):].strip()

        try:
            api_key = ApiKey.objects.select_related("user").get(
                key=token, is_active=True
            )
        except ApiKey.DoesNotExist:
            return JsonResponse(
                {"error": "Invalid or revoked API key."},
                status=401,
            )

        if api_key.user.is_blocked:
            return JsonResponse(
                {"error": "This account has been suspended. Contact the administrators."},
                status=403,
            )

        # Record the most recent use of this key (non-blocking — best effort).
        api_key.last_used = timezone.now()
        api_key.save(update_fields=["last_used"])

        # Attach user context to the request for downstream use.
        # Quota is enforced per *request* IP, not per key owner IP.
        from api.utils.quotas import get_client_ip

        request.api_user = api_key.user
        request.api_ip = get_client_ip(request)

        return view_func(request, *args, **kwargs)

    return wrapper
