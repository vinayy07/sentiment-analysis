!pip install gradio
import gradio as gr
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Perform sentiment analysis on the input text."""
    result = sentiment_analyzer(text)
    return f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.2f})"

# Create a Gradio interface for sentiment analysis
iface = gr.Interface(fn=analyze_sentiment, 
                     inputs="text", 
                     outputs="text", 
                     live=True, 
                     title="Sentiment Analysis",
                     description="Enter some text to analyze its sentiment.")

# Launch the Gradio app
iface.launch()
