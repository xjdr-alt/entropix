from typing import Protocol, TypeVar

import jax

# TODO(qdbp) these type vars would look MUCH less ugly if we just
# bumped to 3.12 for the new non-fugly generics syntax and variance inference

# sampler config typevar
Cfg_contra = TypeVar("Cfg_contra", contravariant=True)  # input only -> contravariant

# sampler state type variable
ST = TypeVar("ST")  # i/o -> invariant


class EntropySampler(Protocol[Cfg_contra, ST]):
  """
  A sampler is any object that can be called to perform a single sampling step (see Sampler.__call__)

  Functions count.
  """

  def __call__(
    self,
    gen_tokens: jax.Array,
    logits: jax.Array,
    attention_scores: jax.Array,
    *,
    cfg: Cfg_contra,
    state: ST | None = None,
    key: jax.Array = jax.random.PRNGKey(1337),
  ) -> tuple[jax.Array, ST]:
    """
    Performs a single sampling step.

    Args:
        gen_tokens: Array of the current token context.
        logits: Array of next token logits predicted by the model
        attention_scores: Array of attention scores are returned by xfmr
        cfg: class-specific configuration object encapsulating any other sampling parameters

    Returns:
        next token as jax.Array
    """
    ...
