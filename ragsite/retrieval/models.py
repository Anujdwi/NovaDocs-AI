from django.db import models

# Create your models here.
# retrieval/models.py
from mongoengine import connect, Document, StringField, IntField, FloatField, DictField
from django.conf import settings

# Connect to MongoDB when module is imported
connect(host=settings.MONGODB_URI)

class UploadedDocument(Document):
    meta = {"collection": "uploaded_documents"}
    title = StringField(required=True)
    file_path = StringField(required=True)
    uploader = StringField()
    extra = DictField()

class Chunk(Document):
    meta = {"collection": "chunks"}
    doc_id = StringField(required=True)   # UploadedDocument.id as str
    chunk_index = IntField(required=True)
    text = StringField(required=True)
    title = StringField()
    vector_id = IntField()   # optional — if you later add vector_id mapping
    score = FloatField()
