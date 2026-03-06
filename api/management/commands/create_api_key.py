"""
Management command: create_api_key

Creates a new API key for a given IP address.  The IP is used to look up or
create the ApiUser record, so quota limits and blocking settings apply as usual.

Usage:
    python manage.py create_api_key --ip 1.2.3.4
    python manage.py create_api_key --ip 1.2.3.4 --label "Lab Python Script"

The full key is printed to stdout exactly once.  Store it securely — it cannot
be recovered afterwards (only the first 10 characters are visible in the admin).
"""

from django.core.management.base import BaseCommand, CommandError

from api.models import ApiKey
from api.utils.quotas import get_or_create_user


class Command(BaseCommand):
    help = "Create a new API key and associate it with the given IP address."

    def add_arguments(self, parser):
        parser.add_argument(
            "--ip",
            required=True,
            help="The IP address to associate with this key (IPv4 or IPv6).",
        )
        parser.add_argument(
            "--label",
            default="",
            help="Optional human-readable label, e.g. 'Lab Python Script'.",
        )

    def handle(self, *args, **options):
        ip = options["ip"].strip()
        label = options["label"].strip()

        # Validate the IP address format by trying to create a GenericIPAddressField value.
        # We rely on Django's own validation here.
        from django.core.validators import validate_ipv46_address
        from django.core.exceptions import ValidationError

        try:
            validate_ipv46_address(ip)
        except ValidationError:
            raise CommandError(f"'{ip}' is not a valid IPv4 or IPv6 address.")

        # Get or create the ApiUser for this IP.
        user = get_or_create_user(ip)

        if user.is_blocked:
            raise CommandError(
                f"The user at {ip} is currently blocked. "
                "Unblock them in the admin before creating a new key."
            )

        # Create the key.
        api_key = ApiKey.objects.create(user=user, label=label)

        # Print the full key — this is the only time it will be shown.
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("API key created successfully!"))
        self.stdout.write("")
        self.stdout.write(f"  Key:    {api_key.key}")
        self.stdout.write(f"  IP:     {user.ip_address}")
        self.stdout.write(f"  Label:  {label or '(none)'}")
        self.stdout.write(f"  Quota:  {user.effective_daily_limit:,} predictions/day")
        self.stdout.write("")
        self.stdout.write(
            self.style.WARNING(
                "⚠  Store this key securely — it will not be shown again."
            )
        )
        self.stdout.write("")
