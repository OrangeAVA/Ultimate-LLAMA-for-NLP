#!/usr/bin/env python
"""
Text Generation using meta-llama/Llama-3.3-70B-Instruct

This script demonstrates how to load the model and tokenizer using Hugging Face's
transformers library and generate text from a given prompt.

Requirements:
    - Python 3.8+
    - transformers (see requirements.txt)
    - torch (see requirements.txt)
    
Note:
    The meta-llama/Llama-3.3-70B-Instruct model is gated and requires authentication.
    Replace 'your_token_here' with your actual Hugging Face token or set to True if you have 
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

def generate_text(prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Generate text using the model.

    Args:
        prompt (str): The input text prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The generated text.
    """
    # Tokenize the input prompt and convert to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text with the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,             # Nucleus sampling for diverse outputs
        repetition_penalty=1.1, # Helps reduce repetitive text
    )
    
    # Decode the generated tokens to a string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # Unique input prompt
    prompt = ("In a futuristic metropolis where neon lights blend with the echoes of forgotten legends, "
              "a solitary inventor unveils a machine that can capture dreams and transform them into reality.")
    
    print("Input Prompt:")
    print(prompt)
    print("\nGenerating text...\n")
    
    # Generate text from the prompt
    result = generate_text(prompt, max_new_tokens=150)
    
    # Print the generated text
    print("Generated Text:\n")
    print(result)

if __name__ == "__main__":
    main()
