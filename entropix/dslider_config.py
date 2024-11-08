from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import math

# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8
VOCAB_SIZE = 128256


@dataclass(frozen=True)
class OutlierThreshold:
  bilinear: jnp.ndarray  # Shape (4, 4)
  linear_state_ent: jnp.ndarray  # Shape (4,)
  linear_state_std: jnp.ndarray  # Shape (4,)
  linear_naked_ent: float
  linear_naked_std: float
  linear_naked_varent: float
  bias: float

  def tree_flatten(self):
    """For JAX pytree handling"""
    arrays = [self.bilinear, self.linear_state_ent, self.linear_state_std]
    aux_data = {
      "linear_naked_ent": self.linear_naked_ent,
      "linear_naked_std": self.linear_naked_std,
      "linear_naked_varent": self.linear_naked_varent,
      "bias": self.bias,
    }
    return arrays, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(
      bilinear=arrays[0],
      linear_state_ent=arrays[1],
      linear_state_std=arrays[2],
      **aux_data,
    )

  def __hash__(self):
    """Static hash implementation"""
    return hash(
      (
        "OutlierThreshold",
        self.bilinear.shape,
        str(self.bilinear.dtype),
        self.linear_state_ent.shape,
        str(self.linear_state_ent.dtype),
        self.linear_state_std.shape,
        str(self.linear_state_std.dtype),
        self.linear_naked_ent,
        self.linear_naked_std,
        self.linear_naked_varent,
        self.bias,
      )
    )


@dataclass(frozen=True)
class ArgmaxThreshold:
  weight: float
  bias: float

  def tree_flatten(self):
    """For JAX pytree handling"""
    aux_data = {"weight": self.weight, "bias": self.bias}
    return [], aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(**aux_data)

  def __hash__(self):
    return hash((self.weight, self.bias))


@dataclass(frozen=True)
class DirichletThreshold:
  weight: float
  bias: float

  def tree_flatten(self):
    """For JAX pytree handling"""
    aux_data = {"weight": self.weight, "bias": self.bias}
    return [], aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(**aux_data)

  def __hash__(self):
    return hash((self.weight, self.bias))


@dataclass(frozen=True)
class TargetEntropy:
  linear: jnp.ndarray  # Shape (4,)
  linear_inv_temp: jnp.ndarray  # Shape (batch_size,)
  bias: float

  def tree_flatten(self):
    arrays = [self.linear, self.linear_inv_temp]
    aux_data = {"bias": self.bias}
    return arrays, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    return cls(linear=arrays[0], linear_inv_temp=arrays[1], bias=aux_data["bias"])

  def __hash__(self):
    """Static hash implementation"""
    return hash(
      (
        "TargetEntropy",
        self.linear.shape,
        str(self.linear.dtype),
        self.linear_inv_temp.shape,
        str(self.linear_inv_temp.dtype),
        self.bias,
      )
    )


@dataclass(frozen=True, eq=True)
class DSConfig:
  # EMWA coefficients
  emwa_logp_base: float
  emwa_logp_exp_factor: float
  emwa_dir_coeff: float
  emwa_temp_coeff: float
  emwa_dir_ent_coeff: float
  emwa_ent_scaffold_coeff: float
  emwa_varent_scaffold_coeff: float
  emwa_ent_naked_coeff: float
  emwa_varent_naked_coeff: float
  emwa_topk_ent_naked_coeff: float

  # Token cross entropy coefficients
  token_cross_ent_scaffold_coeff: float
  token_cross_ent_naked_coeff: float
  token_cross_var_scaffold_coeff: float
  token_cross_var_naked_coeff: float

  # Dirichlet parameters
  perturb_base_coeff: float
  perturb_exp_coeff: float
  """
  dirichlet_support is a subset of the vocabulary of your model.
  recommended tuning:
  1. sample autoregressively conditioned on random hidden state prompts
  2. take the empirical average of logprobs across position and prompts
  3. the support is all logprobs lying above the noise threshold (see normalize_logits in dslider.py)
  """
  dirichlet_support: jnp.ndarray

  # noise floor for logits normalization
  noise_floor: float

  # Threshold parameters
  outlier_threshold: OutlierThreshold
  argmax_threshold: ArgmaxThreshold
  dirichlet_threshold: DirichletThreshold
  target_entropy: TargetEntropy

  # Token outlier
  outlier_topk: int

  def __hash__(self):
    """Static hash implementation that avoids hashing array values"""
    hashable_items = []
    for field in self.__dataclass_fields__.values():
      value = getattr(self, field.name)
      if isinstance(value, (jnp.ndarray, jax.Array)):
        hashable_items.append(hash((str(field.name), value.shape, str(value.dtype))))
      elif isinstance(
        value, (OutlierThreshold, ArgmaxThreshold, DirichletThreshold, TargetEntropy)
      ):
        hashable_items.append(hash(value))
      else:
        hashable_items.append(hash((str(field.name), value)))
    return hash(tuple(hashable_items))

  def tree_flatten(self):
    """Improved flattening for JAX pytree"""
    arrays = []
    aux_data = {}

    for field in self.__dataclass_fields__.values():
      value = getattr(self, field.name)
      if isinstance(value, (jnp.ndarray, jax.Array)):
        arrays.append(value)
      elif isinstance(
        value, (OutlierThreshold, ArgmaxThreshold, DirichletThreshold, TargetEntropy)
      ):
        nested_arrays, nested_aux = value.tree_flatten()
        arrays.extend(nested_arrays)
        aux_data[field.name] = (type(value), nested_aux)
      else:
        aux_data[field.name] = value

    return arrays, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """Improved unflattening for JAX pytree"""
    array_idx = 0
    field_values = {}

    for field_name, field in cls.__dataclass_fields__.items():
      if field_name in aux_data:
        value = aux_data[field_name]
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], type):
          # Reconstruct nested dataclass
          klass, nested_aux = value
          if klass in (OutlierThreshold, TargetEntropy):
            n_arrays = 3 if klass == OutlierThreshold else 2
            nested_arrays = arrays[array_idx : array_idx + n_arrays]
            array_idx += n_arrays
            field_values[field_name] = klass.tree_unflatten(nested_aux, nested_arrays)
          else:
            # For ArgmaxThreshold and DirichletThreshold which have no arrays
            field_values[field_name] = klass(**nested_aux)
        else:
          field_values[field_name] = value
      else:
        field_values[field_name] = arrays[array_idx]
        array_idx += 1

    return cls(**field_values)


register_pytree_node_class(DSConfig)
register_pytree_node_class(OutlierThreshold)
register_pytree_node_class(ArgmaxThreshold)
register_pytree_node_class(DirichletThreshold)
register_pytree_node_class(TargetEntropy)

DEFAULT_DS_CONFIG = DSConfig(
  emwa_logp_base=4.0,
  emwa_logp_exp_factor=3.0,
  emwa_dir_coeff=0.70,
  emwa_temp_coeff=0.70,
  emwa_dir_ent_coeff=0.70,
  emwa_ent_scaffold_coeff=0.70,
  emwa_varent_scaffold_coeff=0.70,
  emwa_ent_naked_coeff=0.70,
  emwa_varent_naked_coeff=0.70,
  emwa_topk_ent_naked_coeff=0.70,
  token_cross_ent_scaffold_coeff=0.65,
  token_cross_ent_naked_coeff=0.65,
  token_cross_var_scaffold_coeff=0.75,
  token_cross_var_naked_coeff=0.65,
  perturb_base_coeff=10.0,
  perturb_exp_coeff=1.0,
  dirichlet_support=jnp.arange(VOCAB_SIZE, dtype=jnp.int32),
  noise_floor=-12.0,
  outlier_threshold=OutlierThreshold(
    bilinear=jnp.ones((4, 4)) * 1.3,
    linear_state_ent=jnp.ones(4) * 0.80,
    linear_state_std=jnp.ones(4) * 0.80,
    linear_naked_ent=1.2,
    linear_naked_std=1.2,
    linear_naked_varent=1.2,
    bias=0.0,
  ),
  argmax_threshold=ArgmaxThreshold(weight=0.1, bias=1.2),
  dirichlet_threshold=DirichletThreshold(weight=0.1, bias=1.2),
  target_entropy=TargetEntropy(
    linear=jnp.array([1.0, 1.0, 1.0, 1.0]), linear_inv_temp=jnp.ones(1) * 8.0, bias=0.0
  ),
  outlier_topk=6,
)
