from entropix.config import SamplerParams, ModelParams, RopeParams
import jax.numpy as jnp
from entropix.generator import generate
from entropix.generator import generate
import tyro
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights


prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""

bp1 = """
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""

prompt2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of Spain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp2 = """
<antThinking>
You're absolutely right. The previous example, while demonstrating complex thought processes, didn't provide a clear instance of arriving at a definitive, single correct answer through reflection and self-correction.
</antThinking>

What is the capital of Spain?<|eot_id|>
"""

prompt3 = """<|start_header_id|>system<|end_header_id|>
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]
<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you retrieve the details for the user with the ID 7890, who has black as their special request?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


bp3 = """
Here is a list of functions in JSON format that I can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]

Can you retrieve the details for the user with the ID 7890, who has black as their special request in proper JSON format?<|eot_id|>

{
  "name": "get_user_info",
  "parameters": {
    "user_id: """

prompt4 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tell me a long and wonderful story about the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Tell me a long and wonderful story about the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp4 = """
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|>

Let me tell you a story about the adventures of the elven mage frieren and her band of heros
Let me tell you a story about the adventures of the elven mage frieren and her band of heros
"""

  
def main():
  
  params = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_words": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "scale_factor": 8,
    "low_freq_factor": 1,
    "high_freq_factor": 4,
    "old_context_len": 8192,
    "use_scaled_rope": True,
    "max_seq_len": 4096
  }
  
  LLAMA_1B_ROPE = RopeParams(
    rope_theta=params["rope_theta"],
    use_scaled_rope=params["use_scaled_rope"],
    scale_factor=params["scale_factor"],
    low_freq_factor=params["low_freq_factor"],
    high_freq_factor=params["high_freq_factor"],
    old_context_len=params["old_context_len"]
  )

  LLAMA_1B_PARAMS = ModelParams(
    n_layers=params["n_layers"],
    n_local_heads=params["n_heads"],
    n_local_kv_heads=params["n_kv_heads"],
    head_dim=params["dim"] // params["n_heads"],
    max_seq_len=params["max_seq_len"],
    rope_params=LLAMA_1B_ROPE,
    d_model=params["dim"]
  )
  
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights()
  #xfmr_weights = load_weights(ckpt_dir=Path('weights/1B-Base'))
  tokenizer = Tokenizer('entropix/tokenizer.model')
  sampler_params = SamplerParams(
    stop_tokens=jnp.load('data/STEER_TOKENS.npy'),
    steer_tokens=jnp.array([128001, 128008, 128009]),
    base_temp=0.666,
    base_top_p=0.90,
    base_top_k=27
  )
  
  sampler_params = SamplerParams(
    stop_tokens=jnp.load('data/STEER_TOKENS.npy'),
    steer_tokens=jnp.array([128001, 128008, 128009]),
    base_temp=0.666,
    base_top_p=0.90,
    base_top_k=27
  )
  
  raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  raw_tokens2 = tokenizer.encode(prompt2, bos=False, eos=False, allowed_special='all')
  raw_tokens3 = tokenizer.encode(prompt3, bos=False, eos=False, allowed_special='all')
  raw_tokens4 = tokenizer.encode(prompt4, bos=False, eos=False, allowed_special='all')

  base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')
  base_raw_tokens2 = tokenizer.encode(bp2, bos=True, eos=False, allowed_special='all')
  base_raw_tokens3 = tokenizer.encode(bp3, bos=True, eos=False, allowed_special='all')
  base_raw_tokens4 = tokenizer.encode(bp4, bos=True, eos=False, allowed_special='all')



  print(prompt)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens1)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens1)
  print('\n')
  print(prompt2)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens2)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens2)
  print('\n')
  print(prompt3)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens3)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens3)
  print('\n')
  print(prompt4)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens4)
  generate(xfmr_weights, model_params, sampler_params, tokenizer, raw_tokens4)
  print('\n')

  #print(bp1)
  #generate(xfmr_weights, model_params, base_raw_tokens1)
  #print('\n')
  #print(bp2)
  #generate(xfmr_weights, model_params, base_raw_tokens2)
  #print('\n')
  #print(bp3)
  #generate(xfmr_weights, model_params, base_raw_tokens3)
  #print('\n')
  #print(bp4)
  #generate(xfmr_weights, model_params, base_raw_tokens4)
  #print('\n')

if __name__ == '__main__':
  tyro.cli(main)