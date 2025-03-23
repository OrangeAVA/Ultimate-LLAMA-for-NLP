#!/usr/bin/env python
"""
Question Answering using meta-llama/Llama-3.3-70B-Instruct

This script demonstrates how to load the model and tokenizer using Hugging Face's
transformers library and generate an answer to a given question.

Requirements:
    - Python 3.8+
    - transformers (see requirements.txt)
    - torch (see requirements.txt)
    
Note:
    The meta-llama/Llama-3.3-70B-Instruct model is gated and requires authentication.
    Replace 'your_token_here' with your actual Hugging Face token or set it to True if you have 
    already authenticated via the Hugging Face CLI (huggingface-cli login).

    Due to the model's size (70B parameters), ensure your hardware (typically a high-memory GPU)
    meets the requirements.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your authentication token here:
auth_token = "your_token_here"  # Replace with your Hugging Face token or set to True if using CLI auth

# Load the tokenizer and model from the Hugging Face model hub with authentication
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", use_auth_token=auth_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", use_auth_token=auth_token)

def answer_question(question: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    """
    Generate an answer for the input question using the model.

    Args:
        question (str): The question to be answered.
        max_new_tokens (int): Maximum number of tokens for the answer.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The generated answer.
    """
    # Construct a QA prompt
    prompt = "Answer the following question concisely:\nQuestion: " + question + "\nAnswer:"
    
    # Tokenize the prompt and convert it to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the answer using the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,             # Nucleus sampling for diverse outputs
        repetition_penalty=1.1, # Helps reduce repetitive text
    )
    
    # Decode the generated tokens to a string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    question = "What are the main challenges in implementing robust cybersecurity measures in modern organizations?"
    
    print("Question:")
    print(question)
    print("\nGenerating answer...\n")
    
    # Generate answer for the provided question
    answer_result = answer_question(question, max_new_tokens=100)
    
    # Print the generated answer
    print("Generated Answer:\n")
    print(answer_result)

if __name__ == "__main__":
    main()
