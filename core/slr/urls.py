from django.urls import path,include
from .views import upload_display_video
urlpatterns = [
   path('', upload_display_video, name='upload_display_video'),
]
