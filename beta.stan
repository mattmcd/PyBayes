data {
  int<lower = 0> N;
  int<lower = 0> trial[N];
  int<lower = 0> success[N];
}
parameters {
  real<lower = 0> alpha0;
  real<lower = 0> beta0;
  real<lower = 0, upper=1> p;
}
model {
  p ~ beta(alpha0, beta0);
  success ~ binomial(trial, p);
}
