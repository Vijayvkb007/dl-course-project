from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView 
from .serializer import ImageSerializer 
from .models import Image
from .utils import prediction

def index(request):
    return HttpResponse("Hello, World!")

class ImageView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request):
        qs_serializer = ImageSerializer(
            data={
                "rgb_image": request.data["rgb_image"],
                "ir_image": request.data["ir_image"]
            },
            context={"request": request}
        )
        
        if qs_serializer.is_valid():
            qs_serializer.save()
            
            return Response({
                "message": "Images uploaded successfully",
                "data": qs_serializer.data, 
            }, status=status.HTTP_200_OK)
            
        return Response({
            "message": qs_serializer.errors,
            "data": None
        }, status=status.HTTP_400_BAD_REQUEST)
        
    def get(self, request):
        qs = Image.objects.all()
        qs_serializer = ImageSerializer(qs, many=True)
        return Response(qs_serializer.data, status=status.HTTP_200_OK)
    