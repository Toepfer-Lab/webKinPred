"""
URL configuration for webKinPred project.
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),

    # Public REST API (v1) — Bearer-token authenticated, no CSRF required.
    # Must be listed before 'api/' so the more-specific prefix matches first.
    path('api/v1/', include('api.urls_v1')),

    # Web UI API — used by the React frontend, CSRF-protected.
    path('api/', include('api.urls')),
]

# Serve static files - using re_path to handle the django_static prefix
urlpatterns += [
    re_path(r'^django_static/(?P<path>.*)$', serve, {
        'document_root': settings.STATIC_ROOT,
    }),
]

# Serve media files
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)