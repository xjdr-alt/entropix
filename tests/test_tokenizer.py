import pytest
from entropix.tokenizer import Tokenizer

def test_tokenizer_encode():
    tokenizer = Tokenizer("entropix/tokenizer.model")
    text = "Hello, world!"
    tokens = tokenizer.encode(text, bos=False, eos=False)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

def test_tokenizer_decode():
    tokenizer = Tokenizer("entropix/tokenizer.model")
    tokens = [1, 2, 3, 4, 5]
    text = tokenizer.decode(tokens)
    assert isinstance(text, str)

def test_tokenizer_special_tokens():
    tokenizer = Tokenizer("entropix/tokenizer.model")
    assert tokenizer.bos_id == tokenizer.special_tokens['']
    assert tokenizer.eos_id == tokenizer.special_tokens['']

def test_tokenizer_encode_decode_roundtrip():
    tokenizer = Tokenizer("entropix/tokenizer.model")
    original_text = "This is a test sentence."
    tokens = tokenizer.encode(original_text, bos=False, eos=False)
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == original_text