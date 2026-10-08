# Summary
Large Language Models (LLMs) can be used for data compression.  I spent some time investigating this, got something that works and is potentially a useful approach for data modelling and LLM interpretability.  This post describes the background, my methodology, results, and interpretation.  

# Background
Data compression can be thought of as turning source data into a shorter sequence such that when a the decoder model is given the shorter sequence it can be used to reconstruct the original data.  LLM auto-regressive token prediction can be thought of as performing a similar task in that it predicts the next token in an output sequence given the previous context.  

Annie Sexton at ngrok wrote a very interesting article [Compression is Prediction](https://ngrok.com/blog/compression-is-prediction) highlighting how next-token prediction and data compression are related.  David MacKay's book 'Information Theory, Inference, and Learning Algorithms' also starts with this approach and Chapter 6 goes into more detail on the [Arithmetic Codes](https://en.wikipedia.org/wiki/Arithmetic_coding) described in the article.  A key point of the article is the LLM logits used to determine the next token can be used to compress the source data using an arithmetic coder.  This can achieve compression ratios much better than classical encoders, at the expense of being much slower.

# Methodology
My laptop is a MacBook Air 2022 M2 with 16GB of RAM so I used [Rapid-MLX](https://rapidmlx.com/) for running models locally.  Going from the blog post to working code was an AI-assisted process using Google AI Pro and Gemini 3.8 Flash, shared as this [transcript](https://share.gemini.google/vMHhOsUMepOI).  This was an iterative process and involved going from the original Arithmetic Coder approach to an [Asymmetric Numeral Systems](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) Encoder as the final outcome.  The code is available in my general playing-around reop at [journal_20260929_llm_entropy_coder.py](https://github.com/mattmcd/PyBayes/blob/master/scripts/journal_20260929_llm_entropy_coder.py).

From the transcript, the Iterative Evolution of the Algorithm: 

* Iteration 1: Baseline 32-bit Arithmetic Coder (Interval Bisection)
  Fixed-point cumulative frequency table; MLX KV-cache prompt steps.
* Iteration 2: 64-bit Streaming rANS (State Machine)
  Upgraded from AC to rANS (single scalar state, LIFO encoding, fast modulo decoding).
* Iteration 3: Metal & MLX Numerical Hardening
  Resolved bfloat16 buffer errors, GPU int64 scatter limits, and tokenizer BOS collisions.
* Iteration 4: Escaped Adaptive Top-K rANS (Noise Floor Elimination)
  Eliminated tail-frequency budget smearing; packed out-of-vocabulary tokens via raw bits.
* Iteration 5: Model Scaling & Runtime Management
  Evaluated 0.6B vs. 4B trade-offs, analyzed mx.compile cache constraints, and enforced Metal memory cleanup.

# Results
Performance on compressing 'The Dunwich Horror' by H.P. Lovecraft

| Algorithm                 | Bytes  | Ratio   | Bits / Char | Memory | bytes/sec |
| ------------------------- | ------ | ------- | ----------- | ------ | --------- |
| Raw UTF-8                 | 100564 | 100.00% | 8.00        | -      | -         |
| bz2 (Burrows-Wheeler - 9) | 33797  | 33.61%  | 2.69        | -      | -         |
| Qwen3.5-9B-4bit           | 13276  | 13.20%  | 1.06        | 4.8GB  | 58        |
| Qwen3-0.6B-4bit           | 18100  | 18.00%  | 1.44        | 319MB  | 148       |
| LFM2.5-1.2B-Instruct-4bit | 16464  | 16.37%  | 1.31        | 638MB  | 242       |
Performance on source code:

| Algorithm                 | Bytes | Ratio   | Bits / Char | Memory | bytes/sec |
| ------------------------- | ----- | ------- | ----------- | ------ | --------- |
| Raw UTF-8                 | 33300 | 100.00% | 8.00        | -      | -         |
| bz2 (Burrows-Wheeler - 9) | 7883  | 23.67%  | 1.89        | -      | -         |
| Qwen3-0.6B-4bit           | 2076  | 6.23%   | 0.50        | 319MB  | 194       |
| LFM2.5-1.2B-Instruct-4bit | 2576  | 7.74%   | 0.62        | 638MB  | 314       |

# Conclusions
The main takeaways from this are that more advanced models generally achieve higher compression, approaching the estimated Shannon limit of ~1bit/char for English text.  This is at the expense of increased memory usage and decreased speed.  Source code compresses better due to its more regular structure.  However, even with the fastest local LLM this is still too slow to be useful for general compression tasks.

As a Data Scientist, I found the compression ratio to be an interesting derived feature that makes this a potentially useful tool.  It is related to the entropy of the source data as well as the entropy of the LLM training data, and the capacity of the LLM.  In general this is related to topics such as anomaly detection where the reconstruction loss of a model can be used to detect occurrences outside the model's training dataset. 

A potential application would be to use an LLM to compress earnings statements from a range of different companies, with a lower compression ratio being a potential indicator of surprising information.  Of course, this is an existing application area of Natural Language Processing so better techniques will be being used already in industry.  