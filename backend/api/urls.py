from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'documents', views.DocumentViewSet, basename='documents')

urlpatterns = [
    path('v1/', include(router.urls)),
    path('v1/chat/', views.chat_view, name='chat'),
    path('v1/chat/<int:session_id>/', views.chat_history, name='chat_history'),
    path('health/',views.health, name='health'),
]