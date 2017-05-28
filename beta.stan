data {
  int<lower = 0> N;
  int<lower = 0> trial[N];
  int<lower = 0> success[N];
}
transformed data {
  real p[N];
  for (i in 1:N)
      p[i] = 1.0*success[i] / trial[i];
}
parameters {
  real<lower = 0> alpha0;
  real<lower = 0> beta0;
}
model {
  p ~ beta(alpha0, beta0);
}
