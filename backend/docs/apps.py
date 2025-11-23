from django.apps import AppConfig
from django.conf import settings
import mongoengine


class DocsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "docs"

    def ready(self):
        #connect to mongo when the app is ready
        mongo_uri = getattr(settings, 'MONGO_URI', None)
        if mongo_uri:
            mongoengine.connect(host=mongo_uri)
