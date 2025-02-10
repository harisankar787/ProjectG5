import requests
from sqlalchemy.orm import Session  # Import Session
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from googleapiclient.discovery import build
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import time
import os
import re
from html import unescape  # Decode HTML entities

import os

api_key = os.getenv("YOUTUBE_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Set up API key and service for YouTube API
if not api_key:
    raise ValueError("API key is missing. Set it using an environment variable.")
youtube = build('youtube', 'v3', developerKey=api_key)

# Load sentiment analysis model and tokenizer
API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"

headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}


# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))  # Use db.session.get()

# Extract YouTube Video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# Fetch YouTube Comments
def get_comments(video_id, max_comments=100):
    comments, next_page_token = [], None
    try:
        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=min(20, max_comments - len(comments)), pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get('items', []):
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                if len(comments) >= max_comments:
                    break
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
            time.sleep(0.5)  # Avoid API rate limits
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

# Preprocess Comments
def preprocess_comment(comment):
    comment = unescape(comment)  # Decode HTML entities
    comment = re.sub(r'<[^>]+>', '', comment)  # Remove HTML tags
    comment = re.sub(r'\d+', '', comment)  # Remove numbers
    comment = re.sub(r'@\w+', '@user', comment)  # Replace mentions
    comment = re.sub(r'http\S+', 'http', comment)  # Replace URLs
    return comment.strip()

# Sentiment Analysis
def analyze_sentiment(comments):
    try:
        if not comments:
            print("No comments to analyze.")
            return {"positive": 0, "neutral": 0, "negative": 0}

        # Initialize counters for each sentiment
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Preprocess comments (if any preprocessing is needed)
        processed_comments = [preprocess_comment(comment) for comment in comments]

        # Send API request
        response = requests.post(API_URL, headers=headers, json={"inputs": processed_comments})

        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return {"positive": 0, "neutral": 0, "negative": 0}

        predictions = response.json()

        if isinstance(predictions, dict) and "error" in predictions:
            print(f"Model error: {predictions['error']}")
            return {"positive": 0, "neutral": 0, "negative": 0}

        # Initialize the sentiment mapping
        label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

        # Loop through each comment's prediction and count the sentiments
        for prediction in predictions:
            # Sort predictions by score in descending order and get the top one
            sorted_predictions = sorted(prediction, key=lambda x: x['score'], reverse=True)
            top_label = sorted_predictions[0]  # Get the label with the highest score
            sentiment = top_label['label']

            # Increment the corresponding sentiment count
            if sentiment == 'LABEL_0':  # Negative
                negative_count += 1
            elif sentiment == 'LABEL_1':  # Neutral
                neutral_count += 1
            elif sentiment == 'LABEL_2':  # Positive
                positive_count += 1

        # Return the counts of each sentiment type
        return {
            "positive": positive_count,
            "neutral": neutral_count,
            "negative": negative_count
        }

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"positive": 0, "neutral": 0, "negative": 0}


# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials.', 'danger')

    return render_template('login.html')

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('index'))

# Dashboard (Enter YouTube URL)
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':  # If user submits a form, redirect to /analyze
        return analyze()
    return render_template('dashboard.html', username=current_user.username)


# Sentiment Analysis Page (Protected)
@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    video_url = request.form.get('video_url')
    if not video_url:
        return render_template('dashboard.html', error="No video URL provided.")

    video_id = extract_video_id(video_url)
    if not video_id:
        return render_template('dashboard.html', error="Invalid YouTube URL.")

    comments = get_comments(video_id)
    if not comments:
        return render_template('dashboard.html', error="No comments found.")

    sentiment_counts = analyze_sentiment(comments)
    total_comments = len(comments)

    return render_template(
        'results.html',
        video_url=video_url,
        positive=round((sentiment_counts["positive"] / total_comments) * 100, 2),
        neutral=round((sentiment_counts["neutral"] / total_comments) * 100, 2),
        negative=round((sentiment_counts["negative"] / total_comments) * 100, 2),
        total_comments=total_comments
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
