from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import DatasetForm, TestDataForm
from .models import Dataset
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from .forms import TestDataForm
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE



# Ensure NLTK data is downloaded
#nltk.download('punkt')
#nltk.download('stopwords')


def index(request):
    return render(request, 'index.html')


def admin_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        if username == "Admin" and password == "admin":
            request.session['admin_logged_in'] = True
            return redirect('admin_home')
        else:
            return render(request, 'admin_login.html', {'error': 'Invalid Username or Password'})
    return render(request, 'admin_login.html')


def admin_home(request):
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')
    return render(request, 'admin_home.html')


def admin_logout(request):
    request.session.flush()
    return redirect('index')


def upload_dataset(request):
    message = ""
    if request.method == "POST":
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            message = "Dataset uploaded successfully!"
        else:
            message = "Failed to upload dataset. Please try again."
    else:
        form = DatasetForm()
    
    return render(request, "upload_dataset.html", {"form": form, "message": message})



# Define the directory where models will be saved
model_dir = os.path.join(os.path.dirname(__file__), 'models')

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def preprocess(request):
    try:
        dataset = Dataset.objects.last()
        if not dataset:
            messages.error(request, "No dataset found. Please upload a dataset first.")
            return redirect('upload_dataset')

        # Read CSV
        df = pd.read_csv(dataset.dataset_file.path)

        # Required columns for sentiment
        required_columns = ['reviews.text', 'sentiment']
        for col in required_columns:
            if col not in df.columns:
                messages.error(request, f"Dataset must contain column: {col}")
                return render(request, "preprocess.html")

        # Fill missing values
        df['reviews.text'] = df['reviews.text'].fillna("")
        df['sentiment'] = df['sentiment'].fillna("neutral")

        # Add primaryCategories column if missing
        if 'categories' not in df.columns:
            df['categories'] = "unknown"
        else:
            df['categories'] = df['categories'].fillna("unknown")

        # Preprocess text
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = nltk.word_tokenize(text)
            tokens = [w for w in tokens if w not in stop_words]
            return " ".join(tokens)

        df['clean_review'] = df['reviews.text'].apply(clean_text)

        # Encode sentiment labels
        le = LabelEncoder()
        df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
        model_dir = os.path.join(settings.BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

        # Save processed dataset
        processed_dir = os.path.join(settings.BASE_DIR, "datasets")
        os.makedirs(processed_dir, exist_ok=True)
        processed_path = os.path.join(processed_dir, "train_preprocessed.csv")
        df.to_csv(processed_path, index=False)

        # Preview for admin
        messages.success(request, "Dataset preprocessing completed successfully!")
        return render(request, "preprocess.html", {
            "preview": df[['reviews.text', 'clean_review', 'sentiment', 'sentiment_encoded', 'categories']].head(10).to_html(classes="table table-bordered")
        })

    except Exception as e:
        messages.error(request, f"Error during preprocessing: {str(e)}")
        return render(request, "preprocess.html")




def build_model(request):
    try:
        dataset = Dataset.objects.last()
        if not dataset:
            messages.error(request, "No dataset found. Please upload and preprocess a dataset first.")
            return redirect('upload_dataset')

        processed_path = os.path.join(settings.BASE_DIR, "datasets", "train_preprocessed.csv")
        if not os.path.exists(processed_path):
            messages.error(request, "Preprocessed dataset file not found. Please preprocess first.")
            return redirect('preprocess')

        df = pd.read_csv(processed_path)

        X = df['clean_review']
        y = df['sentiment_encoded']

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000)
        X_vect = vectorizer.fit_transform(X)

        # Apply SMOTE to balance classes
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_vect, y)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
        svm_model.fit(X_train, y_train)

        # Save model, vectorizer, label encoder
        model_dir = os.path.join(settings.BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(svm_model, os.path.join(model_dir, "svm_model.pkl"))
        joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        le = LabelEncoder()
        le.fit(df['sentiment'])
        joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

        train_acc = accuracy_score(y_train, svm_model.predict(X_train))
        test_acc = accuracy_score(y_test, svm_model.predict(X_test))

        messages.success(request, "Model built successfully!")
        return render(request, "build_model.html", {
            "train_acc": round(train_acc*100, 2),
            "test_acc": round(test_acc*100, 2)
        })

    except Exception as e:
        messages.error(request, f"Error building model: {str(e)}")
        return render(request, "build_model.html")



def user_registration(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            messages.error(request, "All fields are required!")
        elif password != confirm_password:
            messages.error(request, "Passwords do not match!")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
        else:
            User.objects.create_user(username=username, email=email, password=password)
            messages.success(request, "Registration successful! Please login.")
            return redirect("user_login")

    return render(request, "user_registration.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome, {username}!")
            return redirect('user_home')
        else:
            messages.error(request, "Invalid username or password")
    return render(request, "user_login.html")


def user_logout(request):
    logout(request)
    return redirect('index')


@login_required(login_url='user_login')
def user_home(request):
    return render(request, "user_home.html", {"username": request.user.username})



def behavior_analysis_graph(request):
    try:
        dataset_path = os.path.join(settings.BASE_DIR, 'datasets', 'train_preprocessed.csv')
        if not os.path.exists(dataset_path):
            return render(request, "behavior_graph.html", {"error": "No preprocessed dataset found."})

        df = pd.read_csv(dataset_path)

        plt.figure(figsize=(8,5))
        sns.countplot(x='sentiment', data=df, palette='viridis')
        plt.title("Customer Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph = base64.b64encode(image_png).decode('utf-8')

        return render(request, "behavior_graph.html", {"graph": graph})

    except Exception as e:
        return render(request, "behavior_graph.html", {"error": str(e)})



def enter_test_data(request):
    prediction = None
    error = None
    form = TestDataForm()

    try:
        if request.method == "POST":
            form = TestDataForm(request.POST)
            if form.is_valid():
                review_text = form.cleaned_data['review_text']

                # Paths
                model_dir = os.path.join(settings.BASE_DIR, 'models')
                model_path = os.path.join(model_dir, 'svm_model.pkl')
                vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
                label_path = os.path.join(model_dir, 'label_encoder.pkl')

                if not os.path.exists(model_path) or not os.path.exists(vectorizer_path) or not os.path.exists(label_path):
                    error = "Model not found. Please build the model first."
                else:
                    # Load model, vectorizer, label encoder
                    with open(model_path, 'rb') as f:
                        model = joblib.load(f)
                    with open(vectorizer_path, 'rb') as f:
                        vectorizer = joblib.load(f)
                    with open(label_path, 'rb') as f:
                        le = joblib.load(f)

                    # Preprocess input text exactly like training
                    stop_words = set(stopwords.words('english'))
                    review_cleaned = re.sub(r'[^a-z\s]', '', review_text.lower())
                    tokens = nltk.word_tokenize(review_cleaned)
                    tokens = [w for w in tokens if w not in stop_words]
                    review_vector = vectorizer.transform([" ".join(tokens)])

                    # Predict
                    pred_encoded = model.predict(review_vector)[0]
                    prediction = le.inverse_transform([pred_encoded])[0]

    except Exception as e:
        error = str(e)

    return render(request, "enter_test_data.html", {
        "form": form,
        "prediction": prediction,
        "error": error
    })



def build_category_model(request):
    try:
        processed_path = os.path.join(
            settings.BASE_DIR, "datasets", "train_preprocessed.csv"
        )
        if not os.path.exists(processed_path):
            messages.error(request, "Preprocessed dataset not found. Please preprocess first.")
            return redirect('preprocess')

        df = pd.read_csv(processed_path)

        if 'clean_review' not in df.columns or 'categories' not in df.columns:
            messages.error(request, "Dataset must contain 'clean_review' and 'categories' columns.")
            return redirect('preprocess')

        X = df['clean_review']

        def simplify_category(text):
            text = str(text).lower()

            if any(k in text for k in ["book", "ebook", "kindle", "ereader", "novel"]):
                return "Books & E-Readers"
            elif any(k in text for k in ["laptop", "computer", "pc"]):
                return "Computers & Laptops"
            elif any(k in text for k in ["tablet", "ipad"]):
                return "Tablets"
            elif any(k in text for k in ["phone", "mobile", "smartphone", "cell"]):
                return "Mobiles & Accessories"
            elif any(k in text for k in ["speaker", "headphone", "audio", "earbud", "stereo"]):
                return "Audio Devices"
            elif any(k in text for k in ["tv", "television", "streaming", "entertainment", "fire tv"]):
                return "TV & Entertainment"
            elif any(k in text for k in ["alexa", "echo", "assistant", "smart hub", "home automation"]):
                return "Smart Home & Assistants"
            elif any(k in text for k in ["home", "kitchen", "furniture", "appliance"]):
                return "Home & Kitchen"
            elif any(k in text for k in ["clothing", "fashion", "apparel"]):
                return "Clothing & Fashion"
            elif any(k in text for k in ["office", "printer", "scanner"]):
                return "Office Electronics"
            else:
                return "Other"

        df['categories_clean'] = df['categories'].fillna("unknown").apply(simplify_category)
        y = df['categories_clean']

        # Vectorize
        vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
        X_vect = vectorizer.fit_transform(X)

        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

        # Linear SVC
        model = LinearSVC(class_weight='balanced', max_iter=5000)
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

        # Save
        model_dir = os.path.join(settings.BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "category_model.pkl"))
        joblib.dump(vectorizer, os.path.join(model_dir, "category_vectorizer.pkl"))

        messages.success(request, f"Model built successfully! {test_acc*100:.2f}%")
        return render(request, "build_category_model.html", {
            "status": "success",
            "train_acc": round(train_acc*100,2),
            "test_acc": round(test_acc*100,2)
        })

    except Exception as e:
        import traceback
        print("Error in build_category_model:\n", traceback.format_exc())
        messages.error(request, f"Error building category model: {str(e)}")
        return render(request, "build_category_model.html", {"status": "error"})






def recommend_category(request):
    prediction = None
    error = None
    form = TestDataForm()

    try:
        if request.method == "POST":
            form = TestDataForm(request.POST)
            if form.is_valid():
                review_text = form.cleaned_data["review_text"]

                # Use the SAME folder as in build_category_model
                model_dir = os.path.join(settings.BASE_DIR, "models")
                model_path = os.path.join(model_dir, "category_model.pkl")
                vec_path   = os.path.join(model_dir, "category_vectorizer.pkl")

                if not (os.path.exists(model_path) and os.path.exists(vec_path)):
                    error = (
                        "Category model not found. Please build the model first. "
                        f"(Expected: {model_path} and {vec_path})"
                    )
                else:
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vec_path)

                    # Clean text (same style as preprocessing)
                    text = re.sub(r"[^a-z\s]", "", review_text.lower())
                    X_vec = vectorizer.transform([text])

                    prediction = model.predict(X_vec)[0]

    except Exception as e:
        error = str(e)

    return render(request, "recommend_category.html", {
        "form": form,
        "prediction": prediction,
        "error": error
    })
