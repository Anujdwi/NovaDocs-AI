# ingest_demo.py
from django.core.management.base import BaseCommand
from docs.models import DocumentMeta, DocChunk
from uuid import uuid4

class Command(BaseCommand):
    help = "Create a demo document with text chunks (for testing RAG pipeline)."

    def handle(self, *args, **options):
        # create demo document
        doc = DocumentMeta(
            title="Demo Document",
            source="demo",
            metadata={"demo": True}
        )
        doc.save()

        text = """
This is a demo document used for testing the RAG pipeline.
It contains multiple chunks of text that will be embedded, indexed in Chroma, and retrieved during search.
The purpose is to verify that ingestion, chunking, embedding, and retrieval all work end-to-end.
RAG systems retrieve relevant text sections based on user queries.
This demo document simulates a small knowledge base.
        """

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        created = 0
        for p in paragraphs:
            chunk = DocChunk(
                doc=doc,
                chunk_id=str(uuid4()),
                chunk_text=p,
                metadata={"source": "demo"}
            )
            chunk.save()
            created += 1

        self.stdout.write(self.style.SUCCESS(
            f"Demo document created with {created} chunks."
        ))
