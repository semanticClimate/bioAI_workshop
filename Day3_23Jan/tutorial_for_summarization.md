# Stepwise description of Colab notebook used for text summarization

### [Link to the Colab Notebook](https://colab.research.google.com/drive/171C-_HuATlyoby7h0DTVDnUqes5jonfk)

# **Text Summarization Using Transformers on PDF Documents**

This notebook demonstrates how to summarize content from PDF files using a transformer model (`distilbart-cnn-12-6`). The steps involve downloading research papers, extracting text from PDFs, processing the text, and generating summaries for each chunk of text.

## **Prerequisites**

First, make sure to install the necessary dependencies by running the following commands:

```
!pip install transformers[sentencepiece] pymupdf nltk
!pip install pygetpapers
!pip install -- amilib==0.3.9
```

These libraries will allow you to work with transformer models, handle PDFs, process text, and interact with web resources for downloading research papers.

---

## **Step-by-Step Guide**

### **Step 1: Initialize the Model and Tokenizer**

We will use a pre-trained transformer model (`distilbart-cnn-12-6`) for summarization. The model is fine-tuned on CNN/Daily Mail articles and is capable of generating concise summaries for long texts.

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

# Initialize the model and tokenizer
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

Here, we load the `distilbart-cnn-12-6` model along with its tokenizer to encode and decode the text.

---

### **Step 2: Download Research Papers Using `pygetpapers`**

You can use the `pygetpapers` tool to search for academic articles. In this case, we’re querying papers on the medicinal and traditional uses of the Tulsi plant.

```
!pygetpapers --query 'A review medicinal and traditional uses on Tulsi plant (Ocimum sanctum L.)' --pdf --limit 3 --output downloaded_file --save_query
```

This command will download a limited number of papers (3 in this case) and save them in the `downloaded_file` directory.

---

### **Step 3: Convert the PDF Content to HTML for Display**

After downloading the PDF, the next step is to convert it into a user-friendly format. The `amilib` tool is used to convert the data into an HTML table.

```
!amilib HTML --operation DATATABLES --indir downloaded_file
```

This command processes the downloaded file and generates an HTML output.

Now, you can display the HTML content within your notebook using the following code:

```
from IPython.core.display import display, HTML
```

# Path to the HTML file
html_file_path = '/content/downloaded_file/datatables.html'

# Read the HTML file
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Display the HTML content
display(HTML(html_content))
```

This will allow you to visualize the document's content in a table format.

---

### **Step 4: Extract Text from the PDF**

To extract the raw text from the PDF, we’ll use the `PyMuPDF` library.

```
import fitz  # PyMuPDF

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Replace with the path of your uploaded PDF file
pdf_path = "/content/downloaded_file/PMC11678315/fulltext.pdf"
file_content = extract_text_from_pdf(pdf_path)
```

This code will read and extract all the text from the specified PDF document.

---

### **Step 5: Preprocess the Text into Sentences**

Before summarizing the text, we need to split it into sentences. The `nltk` library’s tokenizer can be used for this.

```
import nltk
```

# Split the text into sentences using NLTK
nltk.download('punkt')  # Ensure the necessary tokenizer data is downloaded
sentences = nltk.tokenize.sent_tokenize(file_content)
```

This will tokenize the document into individual sentences for further processing.

---

### **Step 6: Chunk the Text into Manageable Parts**

Most transformer models, including the one we’re using, have a token limit for input length. To ensure we don’t exceed that limit, we split the text into chunks of sentences that the model can handle.

```
# Chunk the text based on token limits of the model
max_chunk_length = tokenizer.model_max_length  # Maximum number of tokens the model can handle
chunks = []
length = 0
chunk = ""

for sentence in sentences:
    sentence_length = len(tokenizer.tokenize(sentence))
    if length + sentence_length <= max_chunk_length:
        chunk += sentence + " "
        length += sentence_length
    else:
        chunks.append(chunk.strip())
        chunk = sentence + " "
        length = sentence_length

# Add the last chunk if there's any left
if chunk:
    chunks.append(chunk.strip())
```

This will break down the text into smaller, manageable chunks that can be fed into the model for summarization.

---

### **Step 7: Generate Summaries for Each Chunk**

Now that the text is split into chunks, we can pass each chunk to the model to generate summaries.

```
# Set a maximum output length for the summary
max_output_length = 50  # You can adjust this value

# Generate summaries for each chunk
summaries = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
    output = model.generate(
        **inputs,
        max_length=max_output_length,
        min_length=10,
        length_penalty=0.8,
        early_stopping=True,
        no_repeat_ngram_size=3  # Ensures less repetition
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove standalone numbers and decimals
    cleaned_output = re.sub(r'\b(?:figure|fig)\s*\d+', '', decoded_output)
    cleaned_output = re.sub(r'\[\d+(?:,\d+)*\]', '', cleaned_output)
    cleaned_output = re.sub(r'\d+(\.\d+)+', '', cleaned_output)

    summaries.append(cleaned_output.strip())
```

For each chunk of text, we generate a summary and clean it up by removing any unnecessary references like figure numbers and citations.

---

### **Step 8: Combine the Summaries into One Final Summary**

Once all the summaries are generated, we combine them into one final summary.

```
# Concatenate all summaries into one paragraph
final_summary = " ".join(summaries)

# Function to display the final summary vertically
def display_summary_vertically(summary, line_width=90):
    print("\n".join([summary[i:i+line_width] for i in range(0, len(summary), line_width)]))

# Display the final summary
display_summary_vertically(final_summary)
```

This will display the final summarized text in a readable format, breaking it into lines if necessary.

---

## **Conclusion**

This notebook demonstrates how to extract, process, and summarize academic content from PDFs using transformer models. By following these steps, you can automate the summarization of large documents, making it easier to digest key insights and findings.

##**Significance of Each Step**
- **Setup Runtime**: Ensures faster computation using GPU.
- **Library Installation:** Provides necessary tools for text extraction and summarization.
- **Model Initialization:** Loads the summarization model and tokenizer.
- **Downloading Research Papers:** Retrieves relevant research PDFs for summarization.
- **Creating DataTables:** Organizes articles for easier exploration.
- **Text Extraction:** Extracts raw text from the PDF for processing.
- **Text Chunking:** Prepares text in model-compatible sizes.
- **Summarization:** Generates concise summaries of the text.
- **Final Summary Display:** Presents the summarized text for review.
"""