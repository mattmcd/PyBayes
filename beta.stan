data {
  int<lower = 0> N;
  real<lower = 0> y[N];
}
parameters {
  real<lower = 0> alpha0;
  real<lower = 0> beta0;
}
model {
  y ~ beta(alpha0, beta0);
}
