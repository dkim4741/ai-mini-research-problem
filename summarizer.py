import gradio as gr
from transformers import pipeline


# Load pre-trained BART model from Hugging Face
summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)


# Count the number of words in a text
def count_words(text):
    return len(text.split())


# Calculate compression ratio between original text and summary
def calculate_compression_ratio(original_text, summary_text):
    original_words = count_words(original_text)
    summary_words = count_words(summary_text)

    if original_words == 0:
        return 0

    return round((summary_words / original_words) * 100, 2)


# Generate summary and basic analysis
def summarize_article(article_text):

    # Check if the input box is empty
    if not article_text.strip():
        return "Please enter a news article."

    word_count = count_words(article_text)

    # Prevent excessively long inputs
    if word_count > 700:
        return "Article is too long. Please use less than 700 words."


    # Generate summary using BART
    summary_result = summarizer(
        article_text,
        max_length=200,
        min_length=100, #or dynamic_min
        do_sample=False
    )

    summary_text = summary_result[0]["summary_text"]

    # Analyze original and summarized text
    original_words = count_words(article_text)
    summary_words = count_words(summary_text)
    compression_ratio = calculate_compression_ratio(
        article_text,
        summary_text
    )

    # Format output for display
    output = (
        f"Original Word Count: {original_words}\n\n"
        f"Summary Word Count: {summary_words}\n\n"
        f"Compression Ratio: {compression_ratio}%\n\n"
        f"Generated Summary:\n"
        f"{summary_text}"
    )

    return output


# Create Gradio web interface
demo = gr.Interface(
    fn=summarize_article,
    inputs=gr.Textbox(
        lines=14,
        placeholder="Paste a long English news article here...",
        label="News Article"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Summary and Analysis"
    ),
    title="Transformer-Based News Summarization Using BART",
    description=(
        "This demo uses a pre-trained BART Transformer model from Hugging Face "
        "to summarize long news articles and analyze the compression ratio."
    )
)


# Launch the web application
if __name__ == "__main__":
    demo.launch()