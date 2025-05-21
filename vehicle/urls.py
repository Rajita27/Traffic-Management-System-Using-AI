from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("video_feed/", views.video_feed, name="video_feed"),
    path("get_vehicle_count/", views.get_vehicle_count, name="get_vehicle_count"),

    path("start/", views.start_detection, name="start_detection"),
    path("stop/", views.stop_detection, name="stop_detection"),
    path("reset/", views.reset_detection, name="reset_detection"),
]
