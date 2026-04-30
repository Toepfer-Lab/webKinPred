from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        # Import tasks to ensure Celery registers them
        import api.tasks  # noqa: F401
        import api.observability.celery_signals  # noqa: F401
