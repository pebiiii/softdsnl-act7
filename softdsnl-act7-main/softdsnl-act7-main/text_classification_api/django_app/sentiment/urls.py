from django.urls import path
from .views import PredictSentiment

urlpatterns = [
    path("predict/", PredictSentiment.as_view(), name="predict"),
]