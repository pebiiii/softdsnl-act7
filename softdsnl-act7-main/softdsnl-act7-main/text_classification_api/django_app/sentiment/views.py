from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model once at startup
model = tf.keras.models.load_model("imdb_text_model.h5")

# Word index for encoding
word_index = imdb.get_word_index()

class PredictSentiment(APIView):
    def post(self, request):
        text = request.data.get("review", "")
        if not text:
            return Response({"error": "No review provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Encode text
        words = text.lower().split()
        encoded = [1]
        for word in words:
            if word in word_index and word_index[word] < 10000:
                encoded.append(word_index[word] + 3)
            else:
                encoded.append(2)
        padded = pad_sequences([encoded], maxlen=200)

        # Predict
        prediction = model.predict(padded)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        return Response({
            "review": text,
            "prediction": sentiment,
            "confidence": float(prediction[0][0])
        })