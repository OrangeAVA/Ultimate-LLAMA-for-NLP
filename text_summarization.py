from transformers import AutoTokenizer, AutoModelForCausalLM

# Set your authentication token here:
auth_token = "your_token_here"  # Replace with your Hugging Face token or set to True if using CLI auth

# Load the tokenizer and model from the Hugging Face model hub with authentication
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E", use_auth_token=auth_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E", use_auth_token=auth_token)

def summarize_text(text: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Generate a summary for the input text using the model.

    Args:
        text (str): The long text to be summarized.
        max_new_tokens (int): Maximum number of tokens for the summary.
        temperature (float): Sampling temperature for diversity.

    Returns:
        str: The generated summary.
    """
    # Construct a summarization prompt
    prompt = "Summarize the following text:\n" + text + "\nSummary:"
    
    # Tokenize the prompt and convert it to PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the summary using the model
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,             # Nucleus sampling for diverse outputs
        repetition_penalty=1.1, # Helps reduce repetitive text
    )
    
    # Decode the generated tokens to a string
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def main():
    # Sample long text to be summarized
    text_to_summarize = (
        "As technology continues to transform every facet of modern society, the need for robust cybersecurity "
        "has never been greater. Companies around the globe are investing heavily in advanced defense mechanisms "
        "to protect their digital assets from evolving cyber threats. This digital arms race has spurred innovation "
        "in encryption techniques, network monitoring, and real-time threat intelligence. With hackers employing "
        "more sophisticated methods every day, maintaining a secure digital environment demands continuous improvement "
        "and vigilance. The stakes are high, as breaches not only compromise sensitive data but also erode public trust and damage reputations."
    )
    
    print("Input Text to Summarize:")
    print(text_to_summarize)
    print("\nGenerating summary...\n")
    
    # Generate summary for the provided text
    summary_result = summarize_text(text_to_summarize, max_new_tokens=100)
    
    # Print the generated summary
    print("Generated Summary:\n")
    print(summary_result)

if __name__ == "__main__":
    main()
