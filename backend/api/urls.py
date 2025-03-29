from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.ImageView.as_view(), name='upload_images'),
]
