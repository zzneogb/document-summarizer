import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class DocumentSummarizer:
    def __init__(self, model: str = None, prompt_file: str = 'chunker_summarizer.agent', provider: str = 'deepseek'):
        """
        Initialize the document summarizer.
        
        Args:
            model: Model to use (default: from environment based on provider)
            prompt_file: Path to file containing summarization prompts
            provider: API provider to use ('openai' or 'deepseek')
        """
        self.provider = provider.lower()
        
        if self.provider == 'openai':
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = "https://api.openai.com/v1"
            self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        else:  # Default to DeepSeek
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.base_url = os.getenv("DEEPSEEK_BASE_URL")
            self.model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        # Load prompts from file
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract system and user prompts from the file
            prompts = {}
            current_prompt = None
            
            for line in content.split('\n'):
                if line.startswith('# System Prompt'):
                    current_prompt = 'system'
                    prompts[current_prompt] = []
                elif line.startswith('# User Prompt Template'):
                    current_prompt = 'user'
                    prompts[current_prompt] = []
                elif current_prompt and line and not line.startswith('#'):
                    prompts[current_prompt].append(line.strip())
            
            self.system_prompt = '\n'.join(prompts.get('system', [])).strip('"\'').strip()
            self.user_prompt_template = '\n'.join(prompts.get('user', [])).strip('"\'').strip()
            
            if not self.system_prompt or not self.user_prompt_template:
                raise ValueError("Could not extract both system and user prompts from the prompt file")
                
        except Exception as e:
            print(f"Warning: Could not load prompts from {prompt_file}: {str(e)}")
            # Fallback to default prompts
            self.system_prompt = (
                "You are a concise summarization assistant. Extract only the key information, actions, "
                "and decisions. Remove all template placeholders, section headers, and any text that's not "
                "part of the actual content. Be direct and to the point. Do not include any meta-commentary, "
                "section numbers, or placeholders like '(Not specified)' or '(retain previous)'. "
                "Only include information that is explicitly stated in the text."
            )
            self.user_prompt_template = (
                "Extract and summarize the key information from this text. Include only the actual content, "
                "no placeholders or template text. Be direct and concise.\n\n{chunk}"
            )
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.max_tokens = 2000  # Max tokens for each summary
        self.context_window = 128000  # Context window size

    def chunk_text(self, text: str, chunk_size: int = 10000) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters,
        trying to break at paragraph boundaries.
        """
        if len(text) <= chunk_size:
            return [text]

        # Try to find a good breaking point near chunk_size
        chunks = []
        while text:
            if len(text) <= chunk_size:
                chunks.append(text)
                break
                
            # Find the last newline within chunk_size
            chunk = text[:chunk_size]
            last_newline = chunk.rfind('\n\n')
            
            if last_newline > 0:
                chunk = text[:last_newline].strip()
                text = text[last_newline:].strip()
            else:
                # If no good breaking point, just split at chunk_size
                chunk = text[:chunk_size]
                text = text[chunk_size:].strip()
            
            chunks.append(chunk)
        
        return chunks

    def summarize_chunk(self, chunk: str, previous_summary: Optional[str] = None) -> str:
        """
        Generate a summary for a chunk of text using the configured provider,
        optionally considering previous context.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if previous_summary:
            messages.append({
                "role": "system",
                "content": f"Previous context (for reference only, do not include in summary):\n{previous_summary}"
            })
        
        # Format the user prompt with the chunk
        user_prompt = self.user_prompt_template.replace("{chunk}", chunk)
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            else:  # DeepSeek
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error generating summary ({self.provider}): {str(e)}")
            return "[Summary generation failed for this chunk]"
            
        # Add a small delay to avoid rate limiting
        time.sleep(1)

    def summarize_document(self, input_path: str, output_path: str, test_mode: bool = False, chunk_offset: int = 0, chunk_limit: int = 20) -> None:
        """
        Summarize a document by breaking it into chunks and summarizing each chunk
        with context from previous chunks.
        
        Args:
            input_path: Path to the input text file
            output_path: Path to save the summary
            test_mode: If True, only process the first few chunks for testing
            chunk_offset: Starting chunk index (0-based)
            chunk_limit: Maximum number of chunks to process
        """
        try:
            # Read the input file
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split the text into chunks
            chunks = self.chunk_text(text)
            print(f"Split document into {len(chunks)} chunks")
            
            # Prepare chunk directory
            chunk_dir = os.path.splitext(output_path)[0] + "_chunks"
            
            # In test mode, only process the first few chunks
            if test_mode:
                test_chunks = 5  # Number of chunks to process in test mode
                print(f"Test mode: Processing only the first {test_chunks} chunks")
                chunks = chunks[:test_chunks]
            else:
                # Process chunks in batches
                start_idx = min(chunk_offset, len(chunks) - 1)
                end_idx = min(chunk_offset + chunk_limit, len(chunks))
                chunks = chunks[start_idx:end_idx]
                print(f"Processing chunks {start_idx + 1}-{end_idx} of {len(chunks) + start_idx}")
            
            # Initialize variables for tracking progress and context
            summaries = []
            previous_summary = None
            
            # Create a directory to store chunks if it doesn't exist
            chunk_dir = os.path.splitext(output_path)[0] + "_chunks"
            os.makedirs(chunk_dir, exist_ok=True)
            
            # Process each chunk with a progress bar
            for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks"), 1):
                # Calculate the actual chunk number in the document
                chunk_num = chunk_offset + i
                
                # Save original chunk to file with actual chunk number
                chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_num:03d}.txt")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                    
                # Print chunk info for debugging
                print(f"\nProcessing chunk {chunk_num} (batch {i}/{len(chunks)}) (size: {len(chunk)} chars)")
                print(f"Original chunk saved to: {chunk_file}")
                
                # Generate summary for the current chunk with previous context
                summary = self.summarize_chunk(chunk, previous_summary)
                
                # Store the summary with actual chunk number and update context for next chunk
                summaries.append(f"--- Chunk {chunk_num} ---\n{summary}\n")
                previous_summary = summary if len(summary) < 1000 else summary[:1000]  # Keep context manageable
                
                # No need to break in test mode as we've already limited the chunks
            
            # Determine if we're appending to an existing file
            file_mode = 'a' if os.path.exists(output_path) and chunk_offset > 0 else 'w'
            
            # Write summaries to the output file
            with open(output_path, file_mode, encoding='utf-8') as f:
                if file_mode == 'w':
                    # Only write header for new files
                    f.write("Document Summary\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Original file: {os.path.basename(input_path)}\n")
                    f.write(f"Total chunks in document: {len(chunks) + chunk_offset}\n")
                    f.write(f"Test mode: {test_mode}\n\n")
                
                # Write batch information
                f.write(f"\n--- Processing chunks {chunk_offset + 1} to {chunk_offset + len(chunks)} ---\n\n")
                f.write("\n".join(summaries))
            
            print(f"\nSummary successfully saved to: {output_path}")
            
        except FileNotFoundError:
            print(f"Error: Input file '{input_path}' not found")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Summarize a document using AI')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('--output', '-o', help='Output file path (default: <input_file>_summary.txt)')
    parser.add_argument('--model', '-m', help='Model to use (overrides provider default)')
    parser.add_argument('--provider', '-P', choices=['openai', 'deepseek'], default='deepseek',
                      help='AI provider to use (default: deepseek)')
    parser.add_argument('--prompt', '-p', default='chunker_summarizer.agent',
                      help='Path to prompt file (default: chunker_summarizer.agent)')
    parser.add_argument('--test', action='store_true',
                      help='Test mode: only process the first few chunks')
    parser.add_argument('--offset', type=int, default=0,
                      help='Start processing from this chunk (0-based)')
    parser.add_argument('--limit', type=int, default=20,
                      help='Maximum number of chunks to process (default: 20)')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_summary.txt"
    
    try:
        summarizer = DocumentSummarizer(
            model=args.model,
            prompt_file=args.prompt,
            provider=args.provider
        )
        print(f"Using {args.provider.upper()} with model: {summarizer.model_name}")
        
        summarizer.summarize_document(
            input_path=args.input_file,
            output_path=args.output,
            test_mode=args.test,
            chunk_offset=args.offset,
            chunk_limit=args.limit
        )
    except Exception as e:
        print(f"Failed to initialize the summarizer: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    main()
