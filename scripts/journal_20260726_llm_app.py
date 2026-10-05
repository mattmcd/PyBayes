# Gemini code

import json
from typing import Literal, Union, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # BlackJAX NUTS sampler seems to have issues with JAX METAL on Apple M2

import pytensor
pytensor.config.cxx = ""
pytensor.config.mode = "NUMBA"
# Override default PyTensor cxxflags to prevent passing -ld64 to Apple Clang
# pytensor.config.gcc__cxxflags = "-O3 -fno-math-errno -Wno-unused-label -Wno-unused-variable -Wno-write-strings"

import cvxpy as cp
import pymc as pm

# -------------------------------------------------------------------
# 1. Define Structured Schemas
# -------------------------------------------------------------------
class ConvexOptParams(BaseModel):
    objective: str = Field(description="Description of objective function, e.g. 'minimize x^2 + y^2'")
    constraints: list[str] = Field(description="List of constraint strings, e.g. ['x + y >= 1']")
    variables: list[str] = Field(description="List of variable names, e.g. ['x', 'y']")


class BayesianParams(BaseModel):
    prior_type: str = Field(description="e.g. Normal, Beta, Uniform")
    likelihood_type: str = Field(description="e.g. Normal, Bernoulli, Poisson")
    observations: list[float] = Field(default_factory=list, description="Data points provided in query")


class RoutingDecision(BaseModel):
    target_library: Literal["cvxpy", "pymc", "unknown"]
    confidence: float
    reasoning: str
    params: Union[ConvexOptParams, BayesianParams, Dict[str, Any]]


# -------------------------------------------------------------------
# 2. Execution Handlers
# -------------------------------------------------------------------
def run_cvxpy_solver(params: ConvexOptParams):
    print("\n[Executing CVXPY Pipeline]")
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    prob = cp.Problem(cp.Minimize(x ** 2 + y ** 2), [x + y >= 1])
    prob.solve()
    return {"status": prob.status, "value": float(prob.value), "x": float(x.value), "y": float(y.value)}


def run_pymc_sampler(params: BayesianParams):
    print("\n[Executing PyMC Pipeline]")
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)
        data = params.observations if params.observations else [1.0, 1.2, 0.9, 1.1]
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data)
        idata = pm.sample(draws=200, tune=200, return_inferencedata=True, progressbar=False,
                          nuts_sampler='nutpie'
                          )

    summary = pm.stats.summary(idata)
    return summary[["mean", "sd"]].to_dict()


# -------------------------------------------------------------------
# 3. Rapid-MLX Router (OpenAI SDK Client)
# -------------------------------------------------------------------
# Connect to local Rapid-MLX instance on default port 8000
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="rapid-mlx"  # Rapid-MLX doesn't require auth, but SDK requires a dummy string
)


def route_and_execute(prompt: str, model_name: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"):
    system_prompt = (
        "You are an expert mathematical intent classifier. "
        "Analyze the user query and classify whether it requires convex optimization (cvxpy) "
        "or Bayesian inference/probabilistic modeling (pymc). "
        "Extract key parameters matching the required schema."
    )

    # Rapid-MLX supports OpenAI's structured outputs via response_format/parse
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=RoutingDecision,
        temperature=0.0
    )

    decision = completion.choices[0].message.parsed

    print(f"Routing Decision: {decision.target_library.upper()}")
    print(f"Reasoning: {decision.reasoning}")

    # Dispatch to appropriate library
    if decision.target_library == "cvxpy":
        cvx_params = ConvexOptParams.model_validate(decision.params)
        return run_cvxpy_solver(cvx_params)
    elif decision.target_library == "pymc":
        mc_params = BayesianParams.model_validate(decision.params)
        return run_pymc_sampler(mc_params)
    else:
        return {"error": "Could not determine appropriate solver."}


# -------------------------------------------------------------------
# 4. Usage Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    opt_query = "Find values of x and y that minimize x^2 + y^2 such that x + y is at least 1."
    bayesian_query = "Fit a normal prior to estimating the mean height from measurements [170, 172, 168, 175]."

    model_name = 'qwen3.5-4b-4bit'

    print("--- Test 1: Convex Optimization ---")
    res1 = route_and_execute(opt_query, model_name=model_name)
    print("Result:", res1)

    print("\n--- Test 2: Bayesian Inference ---")
    res2 = route_and_execute(bayesian_query, model_name=model_name)
    print("Result:", res2)