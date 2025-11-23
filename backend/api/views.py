from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import IngestSerializer
from docs.models import DocumentMeta, DocChunk
from docs.utils.text_extractor import extract_text_from_file
from docs.utils.chunker import chunk_text
from docs.utils.rag import get_rag_engine
from core.models import ChatSession, ChatMessage
import uuid
import os
from django.conf import settings

# Create your views here.

class DocumentViewSet(viewsets.ViewSet):
    def list(self, request):
        docs = DocumentMeta.objects.order_by('-uploaded_at').limit(50)
        results = [{'id': str(d.id), 'title': d.title, 'source': d.source} for d in docs]
        return Response(results)
    
    def retrieve(self, request, pk=None):
        try:
            d = DocumentMeta.objects.get(id=pk)
            data = {'id': str(d.id), 'title': d.title, 'source': d.source}
            return Response(data)
        except DocumentMeta.DoesNotExist:
            return Response({'detail': 'Not Found'}, status=status.HTTP_404_NOT_FOUND)
        
    @action(detail=False, methods=['post'])
    def ingest(self, request):
        """
        Ingest a file (multipart) or return validation error.
        Saves DocumentMeta and chunks into Mongo.
        """
        serializer = IngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        title = serializer.validated_data.get('title')
        file = request.FILES.get('file')
        source_url = serializer.validated_data.get('source_url')

        if not file and not source_url:
            return Response({'detail': 'Provide file or source_url'}, status=status.HTTP_400_BAD_REQUEST)
        
        if file:
            tmp_path = f"/tmp/{uuid.uuid4()}_{file.name}"
            with open(tmp_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)
            text = extract_text_from_file(tmp_path)
            os.remove(tmp_path)
        else:
            # TODO : implement fetch from source_url
            text = ''

        doc = DocumentMeta(title=title or (file.name if file else 'no-title'),
                           source=source_url or 'upload').save()
        chunks = chunk_text(text, chunk_size=400, overlap=50)
        saved = 0
        for c in chunks:
            DocChunk(doc=doc, chunk_text=c['text'], chunk_id=c['chunk_id'], metadata=c['meta']).save()
            saved += 1

        return Response({'document_id': str(doc.id), 'chunks_count': saved})
    
@api_view(['POST'])
def chat_view(request):
    """
    Handle chat queries using RAG pipeline.
    
    Expected payload:
    {
        "session_id": int (optional, creates new if not provided),
        "message": str (required),
        "doc_id": str (optional, filter by document)
    }
    """
    payload = request.data
    message = payload.get('message', '').strip()
    
    if not message:
        return Response(
            {'detail': 'Message is required'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get or create session
    session_id = payload.get('session_id')
    if session_id:
        try:
            session = ChatSession.objects.get(id=session_id)
        except ChatSession.DoesNotExist:
            return Response(
                {'detail': 'Session not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        # Create new session (for now, use a default user or allow anonymous)
        # In production, you'd use: request.user
        from core.models import User
        try:
            default_user = User.objects.first()
            if not default_user:
                # Create a default user if none exists
                default_user = User.objects.create_user(
                    username='default_user',
                    email='default@example.com',
                    password='temp_password'
                )
        except Exception:
            default_user = None
        
        if default_user:
            session = ChatSession.objects.create(user=default_user)
        else:
            return Response(
                {'detail': 'Unable to create session. No users available.'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    # Save user message
    user_message = ChatMessage.objects.create(
        session=session,
        role='user',
        text=message,
        metadatda={}  # Note: using existing typo in model
    )
    
    # Process query through RAG
    try:
        rag_engine = get_rag_engine()
        doc_id = payload.get('doc_id')  # Optional filter
        result = rag_engine.handle_query(message, filter_by_doc_id=doc_id)
        
        # Save assistant response
        # Note: Model has typo 'assisstant' but we'll use what the model expects
        assistant_message = ChatMessage.objects.create(
            session=session,
            role='assisstant',  # Using model's expected value (has typo in model definition)
            text=result['answer'],
            metadatda={
                'sources': result['sources'],
                'chunks_used': result['chunks_used']
            }
        )
        
        return Response({
            'session_id': session.id,
            'answer': result['answer'],
            'sources': result['sources'],
            'chunks_used': result['chunks_used']
        })
    except Exception as e:
        import traceback
        error_details = str(e)
        if hasattr(e, '__traceback__'):
            error_details += f"\n\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        return Response(
            {'detail': f'Error processing query: {error_details}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def chat_history(request, session_id):
    """
    Retrieve chat history for a session.
    GET /api/v1/chat/<session_id>/
    """
    try:
        session = ChatSession.objects.get(id=session_id)
        messages = session.messages.all().order_by('created_at')
        
        history = [{
            'id': msg.id,
            'role': msg.role,
            'text': msg.text,
            'metadata': msg.metadatda or {},
            'created_at': msg.created_at.isoformat()
        } for msg in messages]
        
        return Response({
            'session_id': session.id,
            'started_at': session.started_at.isoformat(),
            'messages': history
        })
    except ChatSession.DoesNotExist:
        return Response(
            {'detail': 'Session not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )

@api_view(['GET'])
def health(request):
    return Response({'status': 'ok'})
