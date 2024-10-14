import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, font
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from keybert import KeyBERT
import torch
import gc
import shutil
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')


class AdvancedLegalDocumentAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Legal Document Analyzer")
        self.geometry("1400x900")

        # Load NLP models
        self.load_nlp_models()

        # Configure grid layout (3x3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar frame with widgets
        self.create_sidebar()

        # Create main content area
        self.create_main_content()

    def load_nlp_models(self):
        # Load Hugging Face model for text simplification
        self.simplification_model_name = "facebook/bart-large-cnn"
        self.simplification_tokenizer = AutoTokenizer.from_pretrained(
            self.simplification_model_name)
        self.simplification_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.simplification_model_name)

        # Load spaCy model for NER
        self.nlp = spacy.load("en_core_web_sm")

        # Load sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()

        # Load keyword extractor
        self.kw_model = KeyBERT()

        # Load summarization pipeline
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn")

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=1, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text="Legal Doc Analyzer", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        buttons = [
            ("Clear Input", self.clear_input),
            ("Upload Image", self.upload_image),
            ("Simplify", self.simplify_document),
            ("Analyze", self.analyze_document),
            ("Summarize", self.summarize_document),
            ("Cleanup Models", self.confirm_cleanup)
        ]

        for i, (text, command) in enumerate(buttons, start=1):
            btn = ctk.CTkButton(self.sidebar_frame, text=text, command=command)
            btn.grid(row=i, column=0, padx=20, pady=10)

    def create_main_content(self):
        # Input area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.input_frame.grid_rowconfigure(1, weight=1)
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.input_label = ctk.CTkLabel(
            self.input_frame, text="Input Text", font=ctk.CTkFont(size=16, weight="bold"))
        self.input_label.grid(row=0, column=0, padx=10,
                              pady=(10, 0), sticky="w")

        self.input_text = ctk.CTkTextbox(self.input_frame, height=300)
        self.input_text.grid(row=1, column=0, padx=10,
                             pady=(5, 10), sticky="nsew")
        self.input_text.insert("1.0", "Enter legal text here...")

        # Analysis output area (moved to middle)
        self.analysis_frame = ctk.CTkFrame(self)
        self.analysis_frame.grid(
            row=0, column=2, padx=20, pady=20, sticky="nsew")
        self.analysis_frame.grid_rowconfigure(1, weight=1)
        self.analysis_frame.grid_columnconfigure(0, weight=1)

        self.analysis_label = ctk.CTkLabel(
            self.analysis_frame, text="Analysis Results", font=ctk.CTkFont(size=16, weight="bold"))
        self.analysis_label.grid(
            row=0, column=0, padx=10, pady=(10, 0), sticky="w")

        self.analysis_text = ctk.CTkTextbox(self.analysis_frame, height=300)
        self.analysis_text.grid(row=1, column=0, padx=10,
                                pady=(5, 10), sticky="nsew")

        # Summary/Simplified output area (moved to bottom)
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=1, column=1, columnspan=2,
                               padx=20, pady=(0, 20), sticky="nsew")
        self.output_frame.grid_rowconfigure(1, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)

        self.output_label = ctk.CTkLabel(
            self.output_frame, text="Output", font=ctk.CTkFont(size=16, weight="bold"))
        self.output_label.grid(row=0, column=0, padx=10,
                               pady=(10, 0), sticky="w")

        self.output_text = ctk.CTkTextbox(self.output_frame, height=300)
        self.output_text.grid(row=1, column=0, padx=10,
                              pady=(5, 10), sticky="nsew")

    def clear_input(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.analysis_text.delete("1.0", tk.END)
        self.output_label.configure(text="Output")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if file_path:
            try:
                extracted_text = self.extract_text_from_image(file_path)
                self.input_text.delete("1.0", tk.END)
                self.input_text.insert(tk.END, extracted_text)
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to process image: {str(e)}")

    def extract_text_from_image(self, image_path):
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text

    def simplify_document(self):
        input_text = self.input_text.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showwarning(
                "Warning", "Please input text or upload an image first.")
            return

        self.output_label.configure(text="Simplified Output")
        simplified_text = self.simplify_legal_document(input_text)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, simplified_text)

    def simplify_legal_document(self, text):
        sentences = sent_tokenize(text)
        simplified_sentences = []
        for sentence in sentences:
            inputs = self.simplification_tokenizer(
                "simplify: " + sentence, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.simplification_model.generate(
                **inputs, max_length=150, num_beams=4, early_stopping=True)
            simplified_sentence = self.simplification_tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            simplified_sentences.append(simplified_sentence)
        return ' '.join(simplified_sentences)

    def analyze_document(self):
        input_text = self.input_text.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showwarning(
                "Warning", "Please input text or upload an image first.")
            return

        analysis = self.perform_nlp_analysis(input_text)
        self.analysis_text.delete("1.0", tk.END)
        self.analysis_text.insert(tk.END, analysis)

    def perform_nlp_analysis(self, text):
        analysis = ""

        # Named Entity Recognition
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        analysis += "Named Entities:\n"
        for entity, label in entities:
            analysis += f"- {entity}: {label}\n"
        analysis += "\n"

        # Sentiment Analysis
        sentiment = self.sia.polarity_scores(text)
        analysis += f"Sentiment Analysis:\n"
        analysis += f"- Positive: {sentiment['pos']:.2f}\n"
        analysis += f"- Neutral: {sentiment['neu']:.2f}\n"
        analysis += f"- Negative: {sentiment['neg']:.2f}\n"
        analysis += f"- Compound: {sentiment['compound']:.2f}\n\n"

        # Keyword Extraction
        keywords = self.kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        analysis += "Top Keywords:\n"
        for keyword, score in keywords:
            analysis += f"- {keyword}: {score:.2f}\n"

        return analysis

    def summarize_document(self):
        input_text = self.input_text.get("1.0", tk.END)
        if not input_text.strip():
            messagebox.showwarning(
                "Warning", "Please input text or upload an image first.")
            return

        self.output_label.configure(text="Summary Output")
        summary = self.summarize_text(input_text)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, summary)

    def summarize_text(self, text):
        max_chunk_length = 1000
        chunks = [text[i:i+max_chunk_length]
                  for i in range(0, len(text), max_chunk_length)]
        summaries = []

        for chunk in chunks:
            summary = self.summarizer(
                chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return " ".join(summaries)

    def confirm_cleanup(self):
        # Show confirmation dialog
        if messagebox.askyesno("Confirm Cleanup", "Are you sure you want to cleanup the models? This will remove them from memory and delete the cache."):
            self.cleanup_models()

    def cleanup_models(self):
        # Unload models
        del self.simplification_tokenizer
        del self.simplification_model
        del self.nlp
        del self.sia
        del self.kw_model
        del self.summarizer

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        # Remove the Hugging Face cache
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        messagebox.showinfo(
            "Cleanup", "Models have been unloaded and cache cleared.")


if __name__ == "__main__":
    app = AdvancedLegalDocumentAnalyzerApp()
    app.mainloop()
