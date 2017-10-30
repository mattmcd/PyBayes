data {
    int<lower = 0> N;
    int<lower = 0> trial[N];
    int<lower = 0> success[N];
}
parameters {
    real <lower=0, upper=1> mu0;
    real mu_t;
    real <lower=0> sigma0;
}
transformed parameters {
    real <lower=0, upper=1> mu[N];
    real <lower = 0> alpha0[N];
    real <lower = 0> beta0[N];
    for (i in 1:N) {
        mu[i] = mu0 + mu_t * log(trial[i]);
        alpha0[i] = mu[i] / sigma0;
        beta0[i] = (1-mu[i])/sigma0;
    }
}
model {
  success ~ beta_binomial(trial, alpha0, beta0);
}
