import torch

def get_clip_text_embedding(text, tokenizer, text_encoder, device):
    """
    Given a text string or a list of text strings, tokenize the text and return the CLIP embedding.
    We use the embedding of the CLS token as our text representation.

    Args:
        text (str or List[str]): The text to encode.

    Returns:
        torch.Tensor: The text embedding of shape (batch_size, hidden_dim).
    """
    # Tokenize the text; the tokenizer automatically pads/truncates
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        # The model returns a sequence of embeddings for all tokens.
        # We take the first token ([CLS]) from the last hidden state as the embedding.
        text_features = text_encoder(**inputs).last_hidden_state  # Shape: (batch, seq_length, hidden_dim)
    return text_features