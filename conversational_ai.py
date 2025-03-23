#!/usr/bin/env python
"""
Conversational AI using meta-llama/Llama-3.3-70B-Instruct

This script demonstrates how to build a simple conversational AI application using Hugging Face's
transformers library and the meta-llama/Llama-3.3-70B-Instruct model. The script loads the pre-trained model,
sets up a conversation loop, and generates responses to user inputs. This is ideal for chatbots, virtual assistants,
or interactive systems requiring dynamic dialogue.

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

def generate_response(conversation_history: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Generate a response for the conversation history using the model.
    
    Args:
        conversation_history (str): The current conversation history including previous turns.
        max_new_tokens (int): Maximum number of tokens for the generated response.
        temperature (float): Sampling temperature for diversity.
        
    Returns:
        str: The generated response text.
    """
    # Construct the conversation prompt
    prompt = (
        "You are a helpful AI assistant engaged in a conversation. Continue the dialogue naturally.\n"
        + conversation_history + "\nAI:"
    )
    
    # Tokenize the prompt and convert it to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the AI response
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
    # Extract the AI response (removing the prompt part)
    response = generated_text.split("AI:")[-1].strip()
    return response

def main():
    print("Welcome to the Conversational AI. Type 'exit' to quit.")
    conversation_history = ""
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Append the user's input to the conversation history
        conversation_history += "\nUser: " + user_input
        
        # Generate an AI response based on the conversation history
        ai_response = generate_response(conversation_history, max_new_tokens=150)
        
        # Print and append the AI response
        print("AI:", ai_response)
        conversation_history += "\nAI: " + ai_response

if __name__ == "__main__":
    main()
