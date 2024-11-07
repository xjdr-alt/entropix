from functools import partial
from typing import NamedTuple, Dict
import jax
import jax.numpy as jnp
from entropix.dslider_config import DSConfig, OutlierThreshold, DirichletThreshold, ArgmaxThreshold, TargetEntropy
from jax.tree_util import register_pytree_node_class
class TuningStats(NamedTuple):
    cross_ent_diff: float
    renyi_div: float
    combined_score: float
    param_gradients: Dict[str, float]

@jax.jit
def renyi_divergence(p: jnp.ndarray, q: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Compute Rényi divergence of order alpha:
    D_α(P||Q) = 1/(α-1) * log(∑ p^α * q^(1-α))

    For α = 1/R where R > 1:
    D_{1/R}(P||Q) = R/(1-R) * log(∑ p^(1/R) * q^(1-1/R))
    """
    # Ensure numerical stability
    p = p + 1e-10
    q = q + 1e-10

    # Normalize if needed
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    q = q / jnp.sum(q, axis=-1, keepdims=True)

    # Compute powers
    p_power = jnp.power(p, alpha)
    q_power = jnp.power(q, 1 - alpha)

    # Compute sum term
    sum_term = jnp.sum(p_power * q_power, axis=-1)

    # Final computation
    return 1.0 / (alpha - 1.0) * jnp.log(sum_term)

class OnlineTuner:
    def __init__(
        self,
        config: DSConfig,
        R: float = 2.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        window_size: int = 100,
    ):
        assert R > 1, "R must be greater than 1"
        self.config = config
        self.R = R
        self.alpha = 1.0 / R  # For Rényi divergence
        self.lr = learning_rate
        self.momentum = momentum
        self.idx = 0
        self.max_idx = 50

        # Initialize parameter momentum buffers
        self.param_momentum = {
            'outlier_bilinear': jnp.zeros_like(config.outlier_threshold.bilinear),
            'outlier_linear_state_ent': jnp.zeros_like(config.outlier_threshold.linear_state_ent),
            'outlier_linear_state_std': jnp.zeros_like(config.outlier_threshold.linear_state_std),
            'dirichlet_weight': jnp.array(0.0),
            'dirichlet_bias': jnp.array(0.0),
            'perturb_base': jnp.array(0.0),
            'perturb_exp': jnp.array(0.0)
        }

        # Statistics tracking
        self.window_size = window_size
        self.stats_buffer = []
        self.total_steps = 0

    def tree_flatten(self):
        """For JAX pytree handling"""
        arrays = [
            self.param_momentum['outlier_bilinear'],
            self.param_momentum['outlier_linear_state_ent'],
            self.param_momentum['outlier_linear_state_std'],
            self.param_momentum['dirichlet_weight'],
            self.param_momentum['dirichlet_bias'],
            self.param_momentum['perturb_base'],
            self.param_momentum['perturb_exp']
        ]
        aux_data = {
            "config": self.config,
            "R": self.R,
            "learning_rate": self.lr,
            "momentum": self.momentum,
            "window_size": self.window_size,
            "stats_buffer": self.stats_buffer,
            "total_steps": self.total_steps
        }
        return arrays, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """For JAX pytree handling"""
        instance = cls(
            config=aux_data["config"],
            R=aux_data["R"],
            learning_rate=aux_data["learning_rate"],
            momentum=aux_data["momentum"],
            window_size=aux_data["window_size"]
        )
        instance.param_momentum = {
            'outlier_bilinear': arrays[0],
            'outlier_linear_state_ent': arrays[1],
            'outlier_linear_state_std': arrays[2],
            'dirichlet_weight': arrays[3],
            'dirichlet_bias': arrays[4],
            'perturb_base': arrays[5],
            'perturb_exp': arrays[6]
        }
        instance.stats_buffer = aux_data["stats_buffer"]
        instance.total_steps = aux_data["total_steps"]
        return instance

    def __hash__(self):
        return hash((
            self.config,
            self.R,
            self.lr,
            self.momentum,
            self.window_size,
            self.total_steps
        ))

    @partial(jax.jit, static_argnames=('self',))
    def compute_metrics(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> TuningStats:
        """Compute current performance metrics using Rényi divergence"""
        # Cross entropy difference
        cross_ent_diff = jnp.mean(token_cross_ent_naked - token_cross_ent_scaffold)

        # Convert logprobs to probs
        scaffold_probs = jnp.exp(scaffold_logprobs)
        naked_probs = jnp.exp(naked_logprobs)

        # Compute Rényi divergence with α = 1/R
        renyi_div = jnp.mean(renyi_divergence(scaffold_probs, naked_probs, self.alpha))

        # Combined score using the same weighting
        combined_score = (1.0/self.R) * cross_ent_diff + ((self.R-1.0)/self.R) * renyi_div

        # Compute gradients for parameters
        param_gradients = self.compute_parameter_gradients(
            scaffold_logprobs, naked_logprobs,
            token_cross_ent_naked, token_cross_ent_scaffold
        )

        return TuningStats(cross_ent_diff, renyi_div, combined_score, param_gradients)

    def get_summary(self) -> str:
        """Generate a summary of tuning statistics"""
        if not self.stats_buffer:
            return "No tuning statistics available"

        recent_stats = self.stats_buffer[-self.window_size:]
        avg_cross_ent = sum(s.cross_ent_diff for s in recent_stats) / len(recent_stats)
        avg_renyi = sum(s.renyi_div for s in recent_stats) / len(recent_stats)
        avg_score = sum(s.combined_score for s in recent_stats) / len(recent_stats)

        return f"""
Online Tuning Summary (R={self.R}, α=1/R={self.alpha:.4f}):
---------------------
Total Steps: {self.total_steps}
Recent Window Statistics (last {len(recent_stats)} steps):
- Average Cross Entropy Difference: {avg_cross_ent:.4f}
- Average Rényi Divergence (α=1/R): {avg_renyi:.4f}
- Average Combined Score: {avg_score:.4f}

Current Parameter Values:
- Outlier Threshold Bilinear: {self.config.outlier_threshold.bilinear.mean():.4f}
- Dirichlet Threshold Weight: {self.config.dirichlet_threshold.weight:.4f}
- Perturbation Base Coefficient: {self.config.perturb_base_coeff:.4f}
"""

    def update(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> DSConfig:
        """Update tuner state and return optimized config"""
        # Compute current metrics and gradients
        stats = self.compute_metrics(
            scaffold_logprobs,
            naked_logprobs,
            token_cross_ent_naked,
            token_cross_ent_scaffold
        )

        # Create new instances instead of using replace
        updated_dirichlet_threshold = DirichletThreshold(
            weight=self.config.dirichlet_threshold.weight + self.lr * self.param_momentum['dirichlet_weight'],
            bias=self.config.dirichlet_threshold.bias + self.lr * self.param_momentum['dirichlet_bias']
        )

        updated_outlier_threshold = OutlierThreshold(
            bilinear=self.config.outlier_threshold.bilinear + self.lr * self.param_momentum['outlier_bilinear'],
            linear_state_ent=self.config.outlier_threshold.linear_state_ent + self.lr * self.param_momentum['outlier_linear_state_ent'],
            linear_state_std=self.config.outlier_threshold.linear_state_std + self.lr * self.param_momentum['outlier_linear_state_std'],
            linear_naked_ent=self.config.outlier_threshold.linear_naked_ent,
            linear_naked_std=self.config.outlier_threshold.linear_naked_std,
            linear_naked_varent=self.config.outlier_threshold.linear_naked_varent,
            bias=self.config.outlier_threshold.bias
        )

        # Update momentum buffers
        for param_name, gradient in stats.param_gradients.items():
            self.param_momentum[param_name] = (
                self.momentum * self.param_momentum[param_name] +
                (1 - self.momentum) * gradient
            )

        # Create new config with updated parameters
        new_config = DSConfig(
            emwa_logp_base=self.config.emwa_logp_base,
            emwa_logp_exp_factor=self.config.emwa_logp_exp_factor,
            emwa_dir_coeff=self.config.emwa_dir_coeff,
            emwa_temp_coeff=self.config.emwa_temp_coeff,
            emwa_dir_ent_coeff=self.config.emwa_dir_ent_coeff,
            emwa_ent_scaffold_coeff=self.config.emwa_ent_scaffold_coeff,
            emwa_varent_scaffold_coeff=self.config.emwa_varent_scaffold_coeff,
            emwa_ent_naked_coeff=self.config.emwa_ent_naked_coeff,
            emwa_varent_naked_coeff=self.config.emwa_varent_naked_coeff,
            emwa_topk_ent_naked_coeff=self.config.emwa_topk_ent_naked_coeff,
            token_cross_ent_scaffold_coeff=self.config.token_cross_ent_scaffold_coeff,
            token_cross_ent_naked_coeff=self.config.token_cross_ent_naked_coeff,
            token_cross_var_scaffold_coeff=self.config.token_cross_var_scaffold_coeff,
            token_cross_var_naked_coeff=self.config.token_cross_var_naked_coeff,
            perturb_base_coeff=self.config.perturb_base_coeff + self.lr * self.param_momentum.get('perturb_base', 0),
            perturb_exp_coeff=self.config.perturb_exp_coeff + self.lr * self.param_momentum.get('perturb_exp', 0),
            dirichlet_support=self.config.dirichlet_support,
            noise_floor=self.config.noise_floor,
            outlier_threshold=updated_outlier_threshold,
            argmax_threshold=self.config.argmax_threshold,
            dirichlet_threshold=updated_dirichlet_threshold,
            target_entropy=self.config.target_entropy,
            outlier_topk=self.config.outlier_topk
        )

        # Update statistics buffer
        self.stats_buffer.append(stats)
        if len(self.stats_buffer) > self.window_size:
            self.stats_buffer.pop(0)
        self.total_steps += 1

        return new_config

    @partial(jax.jit, static_argnames=('self',))
    def compute_parameter_gradients(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Compute gradients for parameters with respect to the objective:
        score = (1/R) * (cross_ent_naked - cross_ent_scaffold) +
                ((R-1)/R) * D_{1/R}(scaffold_logprobs||naked_logprobs)
        """
        # Cross entropy term: (1/R) * (cross_ent_naked - cross_ent_scaffold)
        cross_ent_diff = jnp.mean(token_cross_ent_naked - token_cross_ent_scaffold)
        cross_ent_term = (1.0 / self.R) * cross_ent_diff

        # Renyi divergence term: ((R-1)/R) * D_{1/R}(scaffold||naked)
        scaffold_probs = jnp.exp(scaffold_logprobs)
        naked_probs = jnp.exp(naked_logprobs)
        renyi_term = ((self.R - 1.0) / self.R) * jnp.mean(
            renyi_divergence(scaffold_probs, naked_probs, self.alpha)
        )

        # Total objective
        objective = cross_ent_term + renyi_term

        # Use JAX's grad to compute gradients with respect to each parameter
        gradients = {
            'outlier_bilinear': jax.grad(lambda x: objective)(self.config.outlier_threshold.bilinear),
            'outlier_linear_state_ent': jax.grad(lambda x: objective)(self.config.outlier_threshold.linear_state_ent),
            'outlier_linear_state_std': jax.grad(lambda x: objective)(self.config.outlier_threshold.linear_state_std),
            'dirichlet_weight': jax.grad(lambda x: objective)(self.config.dirichlet_threshold.weight),
            'dirichlet_bias': jax.grad(lambda x: objective)(self.config.dirichlet_threshold.bias),
            'perturb_base': jax.grad(lambda x: objective)(self.config.perturb_base_coeff),
            'perturb_exp': jax.grad(lambda x: objective)(self.config.perturb_exp_coeff)
        }

        return gradients

register_pytree_node_class(OnlineTuner)