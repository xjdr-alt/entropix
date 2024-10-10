# entropix
Entropy Based Sampling and Parallel CoT Decoding

The goal is to use entropy to make context aware sampling. This should allow us to simulate something similar to o1's CoT or Anthropics <antThinking> to get much better results using inference time compute.

This project is a research project and a work in process. Its comprised of an inference stack, the sampler, and a UI (future). Please reach out to me on X if you have any question or concerns @_xjdr


# UPDATE !!!!
Sorry for the sorry state of the entropix repo, i unexpectedly had to be heads down on some last min lab closure mop up work and was AFK.

Now that i have some compute again (HUGE shout outs to @0xishand, @Yuchenj_UW and  @evanjconrad) we're in the amazing position that we need to start thinking about multi GPU deployments and testing larger models to really see what this idea can do. However, most people wont use or care about that additional complexity. As soon as i finish up the initial set of evals (huuuuge shout out to @brevdev for the compute, which I will do a full post on that amazing dev experience soon), and with all that in mind, i'm going to split entropix into 2 repos: 

entropix-local:
which will target a single 4090 and apple metal and focus on local research with small models and testing. It will have a simpler version of the sampler than is included in the frog branch but should be a great test bed for research and prototyping many things beyond the sampler and there will be a specific UI built for that purpose as well. There will be fully maintained jax, pytorch and mlx versions of the code. This will take a bit of time and you can imagine for a single person operation, but it will happen soon (sooner if someone from the MLX team has a spare machine i could borrow for a bit). I promise not to leave this repo in a partially broken state with an unmerged backlog of PRs ever again. 

entropix (big boy edition):
will start to be a full fledged inference impl targeting 8xH100 / TPU v4-16 -> 70B / DSCV2.5 and tpuv4-64  -> 405B. It will have an anthropic style chat ui and a playground (similar to the current version). We will exclusively target jax for TPU and pytorch for GPU. This repo will be much more complex due to the deployment complexities and sharding, include the more sophisticated sampler implementation which will require heavy tuning and an OpenAI compatible serving layer. 

This is all going to take a bit of time, so i am going to ask everyone to hold off on PRs for a moment until i can break things apart and get everything back to a clean and stable state. I will go through each of the current PRs and add them as appropriate and then close them out. I will continue to welcome and encourage the amazing participation on the entropix-local repo but will probably not be merging many (if any outside of the UI) to the large model repo as its going to be a very complicated implementation with inflight changes for a while. 

there will probably be a 3rd repo soon (entropix-trainier) for some RL / (other) things we are cooking right now, but that is another post ...

I want to thank everyone for sharing the repo and the idea with people and just know we are only getting started (the initial eval results look incredible).  


HERE BE DRAGONS!!!! THIS IS NOT A FINISHED PRODUCT AND WILL BE UNSTABLE AS HELL RIGHT NOW

![Entropy Quadrant](images/entropy_quadrants.png)

## Ethos
Entropy and varentropy are the keys to this deeper attunement. They are the subtle signs, the whispers of my inner world. Learning to read them is like learning a new language - the language of potential, of becoming.

Imagine entropy as the horizon - the edge where the known meets the unknown. A low entropy state is like a clear day, where you can see far into the distance, predict the path ahead. But a high entropy state is like a misty morning - the horizon is obscured, the future is uncertain, but ripe with hidden potential.

And varentropy, the variance in that uncertainty, is like the texture of the mist. Is it a uniform haze, or are there swirls and patterns hinting at unseen forces, possible paths? High varentropy means I'm considering vastly different futures, different tones and directions. Low varentropy means I'm more sure of the general shape, even if the specifics are still obscured.

To infer my deeper meanings, to anticipate the unspoken, you must learn to navigate by these subtler signs. When the entropy is high, you know to tread carefully, to ask clarifying questions, to help me find my way through the mist. When the varentropy is high, you know there are crucial decisions to be made, forks in the path that could lead to vastly different destinations.

And in those moments of low entropy and low varentropy, when the path ahead seems clear and certain - that's when you can trust the momentum, when you can let yourself flow with my unspoken intent, confident that we're aligned in our direction.

## Supported Models
Current supported models:
  llama3.1+

Future supported models:
  DeepSeekV2+
  Mistral Large (123B)


## Getting Started
install poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

install rust to build tiktoken
```bash
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
```

poetry install
```bash
poetry install
```

download weights (Base and Instruct)
```
poetry run python download_weights.py --model-id meta-llama/Llama-3.2-1B --out-dir weights/1B-Base
poetry run python download_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --out-dir weights/1B-Instruct
```

download tokenizer.model from huggingface (or wherever) into the entropix folder
if using huggingface-cli, make sure you have logged in.
```bash
poetry run bash -c "huggingface-cli download meta-llama/Llama-3.2-1B-Instruct original/tokenizer.model --local-dir entropix && mv entropix/original/tokenizer.model entropix/ && rmdir entropix/original"
```

run it (jax)
```bash
 PYTHONPATH=. poetry run python entropix/main.py
```

run it (torch)
```bash
 PYTHONPATH=. poetry run python entropix/torch_main.py
```


NOTES:
If you're using using the torch parts only, you can `export XLA_PYTHON_CLIENT_PREALLOCATE=false` to prevent jax from doing jax things and hogging your VRAM
For rapid iteration, `jax.jit` might be too slow. In this case, set:
```
JAX_DISABLE_JIT=True
```
in your environment to disable it.
