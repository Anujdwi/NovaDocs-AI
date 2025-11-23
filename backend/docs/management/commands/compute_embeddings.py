# apps/docs/management/commands/compute_embeddings.py
from django.core.management.base import BaseCommand
from docs.models import DocChunk, DocumentMeta
from docs.utils.embeddings import get_embedding_for_text
from docs.utils.vector_store import get_vector_store
import time

BATCH_SIZE = 20

class Command(BaseCommand):
    help = "Compute embeddings for DocChunk documents missing an embedding and push to ChromaDB."

    def add_arguments(self, parser):
        parser.add_argument(
            '--push-to-chroma',
            action='store_true',
            help='Push embeddings to ChromaDB after computing',
        )

    def handle(self, *args, **options):
        qs = DocChunk.objects.filter(embedding__exists=False)  # mongoengine query
        total = qs.count()
        self.stdout.write(f"Found {total} chunks without embeddings.")
        
        if total == 0:
            self.stdout.write(self.style.SUCCESS("All chunks already have embeddings."))
            return
        
        push_to_chroma = options.get('push_to_chroma', True)
        vector_store = None
        if push_to_chroma:
            try:
                vector_store = get_vector_store()
                self.stdout.write("ChromaDB connection established.")
            except Exception as e:
                self.stderr.write(f"Warning: Could not connect to ChromaDB: {e}")
                self.stderr.write("Continuing without ChromaDB push...")
                push_to_chroma = False
        
        i = 0
        chunks_to_push = []
        
        for chunk in qs:
            try:
                emb = get_embedding_for_text(chunk.chunk_text)
                chunk.embedding = emb
                chunk.save()
                i += 1
                
                # Prepare chunk for ChromaDB
                if push_to_chroma and vector_store:
                    doc = chunk.doc
                    chunks_to_push.append({
                        'chunk_id': chunk.chunk_id,
                        'chunk_text': chunk.chunk_text,
                        'embedding': emb,
                        'metadata': chunk.metadata or {},
                        'doc_id': str(doc.id) if doc else None,
                        'doc_title': doc.title if doc else 'Unknown'
                    })
                    
                    # Push in batches
                    if len(chunks_to_push) >= BATCH_SIZE:
                        try:
                            vector_store.add_chunks(chunks_to_push)
                            self.stdout.write(f"Pushed {len(chunks_to_push)} chunks to ChromaDB...")
                            chunks_to_push = []
                        except Exception as e:
                            self.stderr.write(f"Error pushing to ChromaDB: {e}")
                
                if i % BATCH_SIZE == 0:
                    self.stdout.write(f"Processed {i}/{total}...")
            except Exception as e:
                self.stderr.write(f"Error embedding chunk {chunk.chunk_id}: {e}")
                # optionally sleep for rate-limits
                time.sleep(1)
        
        # Push remaining chunks
        if push_to_chroma and vector_store and chunks_to_push:
            try:
                vector_store.add_chunks(chunks_to_push)
                self.stdout.write(f"Pushed final {len(chunks_to_push)} chunks to ChromaDB.")
            except Exception as e:
                self.stderr.write(f"Error pushing final batch to ChromaDB: {e}")
        
        self.stdout.write(self.style.SUCCESS(f"Embeddings computed for {i} chunks."))
        if push_to_chroma and vector_store:
            stats = vector_store.get_collection_stats()
            self.stdout.write(self.style.SUCCESS(f"ChromaDB now has {stats['total_chunks']} total chunks."))
