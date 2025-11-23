from django.contrib import admin
from mongoengine import Document
from docs.models import DocumentMeta, DocChunk

# Note: MongoDB models (mongoengine) don't work with Django admin directly
# You can create custom admin views or use mongoengine's admin if needed
# For now, we'll document them here

# Register your models here.
# Django admin doesn't support mongoengine models directly.
# Use Django REST Framework API or create custom admin views.
