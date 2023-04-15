import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from tensorflow_probability.substrates import jax as tfp # for random number generation (JAX lacks this feature)

# Refernce to https://github.com/google/jax/issues/13327
def binomial(key, n, p, shape=()):
    return tfp.distributions.Binomial(n, probs=p).sample(
        seed=key,
        sample_shape=shape,
    )

def multinomial(key, n, p, shape=()):
    return tfp.distributions.Multinomial(n, probs=p).sample(
        seed=key,
        sample_shape=shape,
    )

class GaussKernel:
    '''
    Gaussian kernel: continuous random variable
    (NOT FINISHED) Truncated Gaussian kernel: continuous random variable
    '''
    def __init__(self, mu, sigma, lower_bound, upper_bound):
        self.mu_ = mu
        self.sigma_ = sigma
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound

    def sample(self, key):
        '''
        This returns a truncated Gaussian distribution
        '''
        while True:
            x = tfp.distributions.Normal(self.mu_, self.sigma_).sample(seed=key)
            if self.lower_bound_ <= x <= self.upper_bound_:
                return x

    def pdf(self, x: float):
        return jsp.stats.norm.pdf(x, loc=self.mu_, scale=self.sigma_)
    
    def log_pdf(self, x: float):
        return jsp.stats.norm.logpdf(x, loc=self.mu_, scale=self.sigma_)
    
    def cdf(self, x: float):
        return jsp.stats.norm.cdf(x, loc=self.mu_, scale=self.sigma_)

class AitchisonAitkenKernel:
    '''
    (NOT FINISHED) Aitchison-Aitken kernel : discrete random variable
    '''
    def __init__(self, choice, n_choices, top = 0.9):
        self.n_choices_ = n_choices
        self.choice_ = choice
        self.top_ = top

    def sample(self, key):
        pass

    def cdf(self, x: float):
        pass

    def cdf(self, xs: jnp.ndarray):
        pass

    def log_cdf(self, x: float):
        pass

    def prob(self):
        pass

class UniformKernel:
    '''
    (NOT IMPLEMENTED) Uniform kernel : discrete random variable
    Original implementation is incomplete
    '''
    def __init__(self, n_choices):
        self.n_choices_ = n_choices

    def sample(self):
        pass

    def cdf(self, x: float):
        if 0 <= x <= self.n_choices_ - 1:
            return 1.0 / self.n_choices_
        else:
            return 0.0
        
    def cdf(self, xs: jnp.ndarray):
        func = jnp.vmap(self.cdf)
        return func(xs)
        
    def log_cdf(self, x: float):
        return jnp.log(self.cdf(x))

    def prob(self):
        return jnp.array([self.cdf(n) for n in range(self.n_choices_)])
    