import tkinter as tk
from tkinter import filedialog, Text, END
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from string import punctuation
import string  # Import string for punctuation
import matplotlib.pyplot as plt
import nltk

# Download stopwords if you haven't already
nltk.download('stopwords')

# Initialize main window
main = tk.Tk()
main.title("Analysis of Women Safety in Indian Cities Using Machine Learning on Tweets")
main.geometry("1300x1200")

# Global variables
filename = ''
tweets_list = []
clean_list = []
pos, neu, neg = 0, 0, 0

# Upload the file
def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset", title="Select file", filetypes=(("CSV files", "*.csv"),))
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

# Read tweets from file
def read():
    text.delete('1.0', END)
    tweets_list.clear()

    try:
        # Read the CSV file
        train = pd.read_csv(filename, encoding="ISO-8859-1")  # Adjust encoding if needed
        
        # Check if 'text' column exists (lowercase)
        if 'text' not in train.columns:
            text.insert(END, "Error: 'text' column not found in the dataset.\n")
            return
        
        # Reading each tweet and appending it to tweets_list
        for i in range(len(train)):
            tweet = train.iloc[i]['text']  # Accessing the 'text' column (lowercase)
            tweets_list.append(tweet)
            text.insert(END, tweet + "\n")
        
        text.insert(END, "\n\nTotal tweets found in dataset: " + str(len(tweets_list)) + "\n\n\n")
    
    except Exception as e:
        text.insert(END, f"Error reading file: {str(e)}\n")

# Function to clean each tweet
def tweetCleaning(doc):
    tokens = doc.split()  # Split the tweet into tokens (words)
    
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # Remove non-alphabetic tokens and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Join tokens back to form the cleaned sentence
    tokens = " ".join(tokens)
    
    return tokens

# Function to clean all tweets
def clean():
    text.delete('1.0', END)
    clean_list.clear()  # Clear any previous cleaned tweets

    # Loop through all tweets and clean them
    for i in range(len(tweets_list)):
        tweet = tweets_list[i].strip()  # Remove leading/trailing spaces and newlines
        cleaned_tweet = tweetCleaning(tweet.lower())  # Clean the tweet
        clean_list.append(cleaned_tweet)  # Add cleaned tweet to the list
        text.insert(END, cleaned_tweet + "\n")  # Display cleaned tweet

    text.insert(END, "\n\nTotal cleaned tweets: " + str(len(clean_list)) + "\n\n")

# Perform sentiment analysis using TextBlob
def machinelearning():
    global pos, neu, neg
    pos, neu, neg = 0, 0, 0
    text.delete('1.0', END)
    for tweet in clean_list:
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        if polarity <= 0.2:
            neg += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment: NEGATIVE\n")
            text.insert(END, f"Polarity Score: {polarity}\n\n")
        elif 0.2 < polarity <= 0.5:
            neu += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment: NEUTRAL\n")
            text.insert(END, f"Polarity Score: {polarity}\n\n")
        else:
            pos += 1
            text.insert(END, tweet + "\n")
            text.insert(END, "Predicted Sentiment: POSITIVE\n")
            text.insert(END, f"Polarity Score: {polarity}\n\n")
    # Display results
    text.insert(END, f"\n\nPositive Sentiments: {pos}\nNeutral Sentiments: {neu}\nNegative Sentiments: {neg}\n")

# Plot the sentiment graph using Matplotlib
def plot_graph():
    global pos, neu, neg
    sentiments = [pos, neu, neg]
    labels = ['Positive', 'Neutral', 'Negative']
    colors = ['green', 'blue', 'red']
    
    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sentiments, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Sentiment Analysis of Women Safety in Indian Cities')
    plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is drawn as a circle.
    plt.show()

# UI elements
pathlabel = tk.Label(main, text="Upload the dataset file", width=100)
pathlabel.pack()
upload_btn = tk.Button(main, text="Upload", command=upload)
upload_btn.pack()

read_btn = tk.Button(main, text="Read Tweets", command=read)
read_btn.pack()

clean_btn = tk.Button(main, text="Clean Tweets", command=clean)
clean_btn.pack()

ml_btn = tk.Button(main, text="Perform Sentiment Analysis", command=machinelearning)
ml_btn.pack()

graph_btn = tk.Button(main, text="Plot Sentiment Graph", command=plot_graph)
graph_btn.pack()

text = Text(main, height=30, width=150)
text.pack()

main.mainloop()
