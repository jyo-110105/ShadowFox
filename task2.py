import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load your dataset
df = pd.read_csv("X_data.csv")  # Make sure this path is correct

# Print columns for verification
print(df.head())
print(df.columns)

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis using the correct column name
df['scores'] = df['clean_text'].apply(lambda tweet: sid.polarity_scores(str(tweet)))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

# Classify sentiment
df['sentiment'] = df['compound'].apply(
    lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral')
)

# Count of each sentiment
sentiment_counts = df['sentiment'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_counts)

# Plot
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Analysis on X Posts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
