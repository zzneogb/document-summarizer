# Document Summarizer

A command-line tool for summarizing large text documents using OpenAI's language models. The tool intelligently chunks large documents and maintains context between chunks for coherent summarization.

## Features

- Processes large text documents by splitting them into manageable chunks
- Maintains context between chunks for coherent summaries
- Uses OpenAI's GPT models for high-quality summarization
- Progress tracking with a progress bar
- Preserves document structure and key information

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Basic usage:
```bash
python summarizer.py path/to/your/document.txt
```

Specify an output file:
```bash
python summarizer.py input.txt -o output_summary.txt
```

Use a different OpenAI model (default is gpt-3.5-turbo):
```bash
python summarizer.py input.txt --model gpt-4
```

## How It Works

1. The document is split into chunks based on paragraph boundaries
2. Each chunk is processed with context from the previous chunk's summary
3. The model generates a concise summary for each chunk
4. All summaries are combined into a single output file

## Output Format

The output file will contain:
- A header with the original filename and number of chunks
- Each chunk's summary, clearly labeled with its chunk number

## Notes

- For large documents, processing may take some time and consume OpenAI API credits
- The quality of summaries depends on the chosen model
- The tool maintains context between chunks but may still lose some document-level coherence

## License

MIT
