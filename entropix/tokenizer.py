import os
from logging import getLogger
from pathlib import Path
from typing import (
  AbstractSet,
  cast,
  Collection,
  Dict,
  Iterator,
  List,
  Literal,
  Optional,
  Sequence,
  Union,
)
import re
import tiktoken
import base64
import tempfile
import requests
import hashlib
import uuid

logger = getLogger(__name__)


# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000

def read_file(blobpath: str) -> bytes:
    if not blobpath.startswith("http://") and not blobpath.startswith("https://"):
        with open(blobpath, "rb") as f:
            return f.read()
    # Avoid using blobfile for public files to help avoid auth issues
    resp = requests.get(blobpath)
    resp.raise_for_status()
    return resp.content

def read_file_cached(blobpath: str) -> bytes:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "data-gym-cache")
    if cache_dir == "":
        # disable caching
        return read_file(blobpath)

    cache_key = hashlib.sha1(blobpath.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    contents = read_file(blobpath)

    os.makedirs(cache_dir, exist_ok=True)
    tmp_filename = cache_path + "." + str(uuid.uuid4()) + ".tmp"
    with open(tmp_filename, "wb") as f:
        f.write(contents)
    os.rename(tmp_filename, cache_path)

    return contents

def load_tiktoken_bpe(tiktoken_bpe_file: str) -> dict[bytes, int]:
       try:
           contents = read_file_cached(tiktoken_bpe_file)
       except Exception as e:
           logger.error(f"Failed to read '{tiktoken_bpe_file}': {e}")
           raise

       bpe_dict = {}
       for line_number, line in enumerate(contents.splitlines(), start=1):
           if not line.strip():
               continue  # Skip empty lines
           parts = line.split()
           if len(parts) != 2:
               logger.error(f"Line {line_number} in {tiktoken_bpe_file} is malformed: '{line}'")
               raise ValueError(f"Line {line_number} in {tiktoken_bpe_file} does not have exactly two elements.")
           token, rank = parts
           try:
               decoded_token = base64.b64decode(token)
               rank_int = int(rank)
           except Exception as e:
               logger.error(f"Error processing line {line_number}: '{line}' - {e}")
               raise
           bpe_dict[decoded_token] = rank_int
       
       return bpe_dict

class Tokenizer:
  """
  Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
  """

  special_tokens: Dict[str, int]

  num_reserved_special_tokens = 256

  pat_str = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
  
  def __init__(self, model_path: str):
    """
    Initializes the Tokenizer with a Tiktoken model.

    Args:
        model_path (str): The path to the Tiktoken model file.
    """
    assert os.path.isfile(model_path), model_path

    
    
    mergeable_ranks = load_tiktoken_bpe(model_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      '<|begin_of_text|>',
      '<|end_of_text|>',
      '<|reserved_special_token_0|>',
      '<|reserved_special_token_1|>',
      '<|finetune_right_pad_id|>',
      '<|step_id|>',
      '<|start_header_id|>',
      '<|end_header_id|>',
      '<|eom_id|>',  # end of message
      '<|eot_id|>',  # end of turn
      '<|python_tag|>',
    ]
    reserved_tokens = [
      f'<|reserved_special_token_{2 + i}|>'
      for i in range(self.num_reserved_special_tokens - len(special_tokens))
    ]
    special_tokens = special_tokens + reserved_tokens

    self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    self.model = tiktoken.Encoding(
      name=Path(model_path).name,
      pat_str=self.pat_str,
      mergeable_ranks=mergeable_ranks,
      special_tokens=self.special_tokens,
    )

    self.n_words: int = num_base_tokens + len(special_tokens)
    # BOS / EOS token IDs
    self.bos_id: int = self.special_tokens['<|begin_of_text|>']
    self.eos_id: int = self.special_tokens['<|end_of_text|>']
    self.eot_id: int = self.special_tokens['<|eot_id|>']
    self.eom_id: int = self.special_tokens['<|eom_id|>']
    self.python_tag_id = self.special_tokens['<|python_tag|>']
    self.pad_id: int = self.special_tokens['<|finetune_right_pad_id|>']
    self.stop_tokens = [
      self.special_tokens['<|eom_id|>'],
      self.special_tokens['<|eot_id|>'],
    ]

  def encode(
    self,
    s: str,
    *,
    bos: bool,
    eos: bool,
    allowed_special: Optional[Union[Literal['all'], AbstractSet[str]]] = None,
    disallowed_special: Union[Literal['all'], Collection[str]] = (),
  ) -> List[int]:
    """
    Encodes a string into a list of token IDs.

    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.
        allowed_tokens ("all"|set[str]): allowed special tokens in string
        disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

    Returns:
        list[int]: A list of token IDs.

    By default, setting disallowed_special=() encodes a string by ignoring
    special tokens. Specifically:
    - Setting `disallowed_special` to () will cause all text corresponding
      to special tokens to be encoded as natural text (insteading of raising
      an error).
    - Setting `allowed_special` to "all" will treat all text corresponding
      to special tokens to be encoded as special tokens.
    """
    if allowed_special is None:
      allowed_special = set()
    assert isinstance(s, str)

    substrs = (
      substr
      for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
      for substr in self._split_whitespaces_or_nonwhitespaces(
        s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
      )
    )
    t: List[int] = []
    for substr in substrs:
      t.extend(
        self.model.encode(
          substr,
          allowed_special=allowed_special,
          disallowed_special=disallowed_special,
        )
      )
    if bos:
      t.insert(0, self.bos_id)
    if eos:
      t.append(self.eos_id)
    return t

  def decode(self, t: Sequence[int]) -> str:
    """
    Decodes a list of token IDs into a string.

    Args:
        t (List[int]): The list of token IDs to be decoded.

    Returns:
        str: The decoded string.
    """
    # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
    return self.model.decode(cast(List[int], t))

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]