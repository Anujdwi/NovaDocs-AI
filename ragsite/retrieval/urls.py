# retrieval/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_file, name="upload_file"),
    path("index_document/", views.index_document, name="index_document"),
    path("query/", views.query_view, name="query_view"),
]
