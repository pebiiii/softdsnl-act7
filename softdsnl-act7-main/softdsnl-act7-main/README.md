# 📝 Activity 7: Text Classification API with TensorFlow + Django (IMDB Reviews)

## 🎯 Objective
In this activity, you will build a **text classification model** with TensorFlow (using the IMDB reviews dataset)  
and then serve it through a **Django REST API**.  

By the end of this activity, you should be able to:
- Train and save a **sentiment analysis model**.  
- Build a **Django app** that loads the model.  
- Send text input via **Postman** and receive predictions in JSON.  

---

## 📂 Folder Structure
```
text_classification_api/
│── requirements.txt
│── train_model.py
│── imdb_text_model.h5       # (generated after training)
│── django_app/
    │── manage.py
    │── django_app/
    │   │── settings.py
    │   │── urls.py
    │   │── wsgi.py
    │── sentiment/
        │── __init__.py
        │── views.py
        │── urls.py
```

---

## 📦 requirements.txt
```
tensorflow
numpy
django
djangorestframework
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Step 1: Train the Model
Create `train_model.py`:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# 2. Pad sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# 3. Build model
model = models.Sequential([
    layers.Embedding(10000, 32, input_length=200),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# 6. Save
model.save("imdb_text_model.h5")
print("✅ Model saved as imdb_text_model.h5")
```

Run:
```bash
python train_model.py
```

---

## 🚀 Step 2: Setup Django Project
Create the Django project:

```bash
django-admin startproject django_app
cd django_app
python manage.py startapp sentiment
```

Add `rest_framework` and `sentiment` to `INSTALLED_APPS` in `settings.py`.

---

## 🚀 Step 3: Create the Sentiment API
Inside `sentiment/views.py`:

```python
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
```

Create `sentiment/urls.py`:

```python
from django.urls import path
from .views import PredictSentiment

urlpatterns = [
    path("predict/", PredictSentiment.as_view(), name="predict"),
]
```

Edit `django_app/urls.py`:

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("sentiment.urls")),
]
```

Run the server:
```bash
python manage.py runserver
```

---

## 🚀 Step 4: Test with Postman
1. Open Postman.  
2. Send a **POST** request to:
   ```
   http://127.0.0.1:8000/api/predict/
   ```
3. In **Body → raw → JSON**, send:
   ```json
   {
     "review": "This movie was fantastic and I loved it"
   }
   ```
4. You should get a response like:
   ```json
   {
     "review": "This movie was fantastic and I loved it",
     "prediction": "Positive",
     "confidence": 0.912345
   }
   ```

---

## 📝 Deliverables
Submit:
1. A **PDF report** named `SOFTDSNL_TextClassification_Surnames.pdf` containing:
   - Screenshots of training results.  
   - At least **5 Postman test results** with reviews of your choice.  
   - A short reflection on the model’s accuracy.  
2. A **GitHub repo link** with your project.

---

## ✅ Grading Criteria
| Criteria                               | Points |
|----------------------------------------|--------|
| Model trains successfully              | 20 pts |
| Django API works (endpoint responds)   | 20 pts |
| 5 Postman test results (screenshots)   | 30 pts |
| Reflection on results                  | 20 pts |
| Clean, working repo submission         | 10 pts |
| **Total**                              | 100 pts |

---
