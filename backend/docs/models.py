from django.db import models
import datetime
from mongoengine import Document, StringField, DateTimeField, DictField, ListField, FloatField, ReferenceField

# Create your models here.
class DocumentMeta(Document):
    title = StringField(required=True)
    source = StringField()
    metadata = DictField()
    uploaded_at = DateTimeField(default=datetime.datetime.utcnow)
    tags = ListField(StringField())

class DocChunk(Document):
    doc = ReferenceField(DocumentMeta)
    chunk_text = StringField()
    chunk_id = StringField(required=True, unique=True)
    metadata = DictField()
    embedding = ListField(FloatField())  # optional if we store embeddings here
    created_at = DateTimeField(default=datetime.datetime.utcnow)