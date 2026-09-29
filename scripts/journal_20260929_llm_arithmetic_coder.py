import struct
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import gc

# Benchmark for comparison
import bz2
import lzma
import zlib
import time


def benchmark_compression(original_text: str, llm_compressed_bytes: bytes):
    raw_bytes = original_text.encode("utf-8")
    raw_size = len(raw_bytes)

    # Standard library codecs at maximum compression levels
    zlib_compressed = zlib.compress(raw_bytes, level=9)
    bz2_compressed = bz2.compress(raw_bytes, compresslevel=9)
    lzma_compressed = lzma.compress(raw_bytes, preset=9)

    results = [
        ("Raw UTF-8", raw_size, 1.0, 8.0),
        (
            "zlib (Deflate - 9)",
            len(zlib_compressed),
            len(zlib_compressed) / raw_size,
            (len(zlib_compressed) * 8) / len(original_text),
        ),
        (
            "bz2 (Burrows-Wheeler - 9)",
            len(bz2_compressed),
            len(bz2_compressed) / raw_size,
            (len(bz2_compressed) * 8) / len(original_text),
        ),
        (
            "lzma (XZ - 9)",
            len(lzma_compressed),
            len(lzma_compressed) / raw_size,
            (len(lzma_compressed) * 8) / len(original_text),
        ),
        (
            "LLM + Arithmetic Coder",
            len(llm_compressed_bytes),
            len(llm_compressed_bytes) / raw_size,
            (len(llm_compressed_bytes) * 8) / len(original_text),
        ),
    ]

    print("\n" + "=" * 70)
    print(f"{'Algorithm':<26} | {'Bytes':>8} | {'Ratio':>10} | {'Bits / Char':>12}")
    print("-" * 70)
    for name, size, ratio, bpc in results:
        print(f"{name:<26} | {size:>8} | {ratio:>9.2%} | {bpc:>12.2f}")
    print("=" * 70)


# Base class for garbage collection
class MLXCoder:

    def __init__(self, model_id: str = "mlx-community/LFM2.5-1.2B-Instruct-4bit"):
        print(f"Loading {model_id}...")
        self.model, self.tokenizer = load(model_id)
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None) or 1

    def close(self):
        """Explicitly release model weights, tokenizer, and pooled Metal buffers."""
        # 1. Sever Python references
        self.model = None
        self.tokenizer = None

        # 2. Force cyclic garbage collection
        gc.collect()

        # 3. Release unified memory blocks back to macOS / Metal driver
        if mx.metal.is_available():
            mx.clear_cache()
            mx.reset_peak_memory()

        print("Model unloaded and Metal cache cleared.")

    # Enable 'with MLXRANSCoder(...) as compressor:' syntax
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def print_memory_usage():
    # Check memory usage
    print(
        f"Active Metal Memory: {mx.get_active_memory() / (1024 ** 2):.1f} MB"
    )
    print(f"Cache Metal Memory : {mx.get_cache_memory() / (1024 ** 2):.1f} MB")


# =====================================================================
# 1. 32-Bit Fixed-Point Arithmetic (Range) Coder
# =====================================================================

class ArithmeticEncoder:
    """Finite-precision 32-bit range encoder."""

    def __init__(self):
        self.precision = 32
        self.top = 1 << self.precision
        self.half = self.top >> 1
        self.quarter = self.top >> 2
        self.three_quarters = self.half + self.quarter

        self.low = 0
        self.high = self.top - 1
        self.underflow_bits = 0
        self.output_bytes = bytearray()
        self._current_byte = 0
        self._bit_count = 0

    def _write_bit(self, bit: int):
        self._current_byte = (self._current_byte << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self.output_bytes.append(self._current_byte)
            self._current_byte = 0
            self._bit_count = 0

    def _emit_bit_and_underflow(self, bit: int):
        self._write_bit(bit)
        opposite_bit = 1 - bit
        while self.underflow_bits > 0:
            self._write_bit(opposite_bit)
            self.underflow_bits -= 1

    def encode(self, low_count: int, high_count: int, total_count: int):
        """Encode a symbol interval [low_count, high_count) out of total_count."""
        range_size = self.high - self.low + 1
        self.high = self.low + (range_size * high_count // total_count) - 1
        self.low = self.low + (range_size * low_count // total_count)

        while True:
            if self.high < self.half:
                self._emit_bit_and_underflow(0)
            elif self.low >= self.half:
                self._emit_bit_and_underflow(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.underflow_bits += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break

            self.low = (self.low << 1) & (self.top - 1)
            self.high = ((self.high << 1) | 1) & (self.top - 1)

    def finish(self) -> bytes:
        """Flush remaining state to byte stream."""
        self.underflow_bits += 1
        if self.low < self.quarter:
            self._emit_bit_and_underflow(0)
        else:
            self._emit_bit_and_underflow(1)

        if self._bit_count > 0:
            self.output_bytes.append(self._current_byte << (8 - self._bit_count))
            self._bit_count = 0

        return bytes(self.output_bytes)


class ArithmeticDecoder:
    """Finite-precision 32-bit range decoder."""

    def __init__(self, data: bytes):
        self.precision = 32
        self.top = 1 << self.precision
        self.half = self.top >> 1
        self.quarter = self.top >> 2
        self.three_quarters = self.half + self.quarter

        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

        self.low = 0
        self.high = self.top - 1
        self.value = 0

        # Prime the decoder with the first 32 bits
        for _ in range(self.precision):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        if self.byte_pos < len(self.data):
            bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.bit_pos = 0
                self.byte_pos += 1
            return bit
        return 0

    def get_current_count(self, total_count: int) -> int:
        """Find the cumulative count corresponding to current internal value."""
        range_size = self.high - self.low + 1
        count = ((self.value - self.low + 1) * total_count - 1) // range_size
        return int(count)

    def decode(self, low_count: int, high_count: int, total_count: int):
        """Advance internal interval window after symbol identification."""
        range_size = self.high - self.low + 1
        self.high = self.low + (range_size * high_count // total_count) - 1
        self.low = self.low + (range_size * low_count // total_count)

        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.value -= self.half
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarters:
                self.value -= self.quarter
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break

            self.low = (self.low << 1) & (self.top - 1)
            self.high = ((self.high << 1) | 1) & (self.top - 1)
            self.value = ((self.value << 1) | self._read_bit()) & (self.top - 1)


# =====================================================================
# 2. Probability Quantization & Model Wrapper
# =====================================================================

def logits_to_discrete_cdf(logits_np: np.ndarray, total_bits: int = 14) -> tuple[np.ndarray, int]:
    """
    Quantize model logits to an integer cumulative frequency table.
    Uses 14-bit (16384) max scale to guarantee total_count * range_size
    fits comfortably inside 32-bit arithmetic ranges without overflow.
    """
    # Numerically stable float64 softmax
    logits_64 = logits_np.astype(np.float64)
    logits_64 -= np.max(logits_64)
    exp_logits = np.exp(logits_64)
    probs = exp_logits / np.sum(exp_logits)

    target_total = 1 << total_bits
    # Ensure every single token has frequency >= 1 to prevent division by zero
    freqs = np.maximum((probs * target_total).astype(np.int64), 1)

    # Build discrete prefix sums
    cum_freq = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum_freq[1:])
    total_count = int(cum_freq[-1])

    return cum_freq, total_count


class MLXArithmeticCoder(MLXCoder):
    def __init__(self, model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        super().__init__(model_id)

    def _next_logits(self, token_id: int, cache) -> np.ndarray:
        inp = mx.array([[token_id]])
        out = self.model(inp, cache=cache)
        mx.eval(out)
        # Squeeze batch & sequence dim: [vocab_size]
        # Squeeze and cast to float32 BEFORE passing across the buffer boundary
        logits_mlx = out[0, -1].astype(mx.float32)
        return np.array(logits_mlx)

    def compress(self, text: str) -> bytes:
        # Disable automatic prefixing of BOS / special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return b""

        encoder = ArithmeticEncoder()
        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id

        for target_token in tokens:
            logits = self._next_logits(current_token, cache)
            cum_freq, total_count = logits_to_discrete_cdf(logits)

            low = int(cum_freq[target_token])
            high = int(cum_freq[target_token + 1])
            encoder.encode(low, high, total_count)

            current_token = target_token

        bitstream = encoder.finish()
        # Pack 4-byte big-endian token count header so decoder knows when to stop
        header = struct.pack(">I", len(tokens))
        return header + bitstream

    def decompress(self, payload: bytes) -> str:
        if len(payload) < 4:
            return ""

        (num_tokens,) = struct.unpack(">I", payload[:4])
        bitstream = payload[4:]

        decoder = ArithmeticDecoder(bitstream)
        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id
        decoded_tokens = []

        for _ in range(num_tokens):
            logits = self._next_logits(current_token, cache)
            cum_freq, total_count = logits_to_discrete_cdf(logits)

            val = decoder.get_current_count(total_count)
            # Binary search to find token matching [cum_freq[k], cum_freq[k+1])
            token_id = int(np.searchsorted(cum_freq, val, side="right") - 1)

            low = int(cum_freq[token_id])
            high = int(cum_freq[token_id + 1])
            decoder.decode(low, high, total_count)

            decoded_tokens.append(token_id)
            current_token = token_id

        return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)


# More sophisticated approach: Asymmetric Numeral Systems (ANS) coder

# =====================================================================
# 1. High-Performance CPU Frequency Table Builder
# =====================================================================

# Vocabulary size for LFM2.5 is ~131,072.
# We set SCALE_BITS = 18 -> M = 262,144, ensuring M > V.
SCALE_BITS = 18
M = 1 << SCALE_BITS


def fast_logits_to_tables(
        logits_mlx: mx.array,
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Bring logits to CPU float32
    logits = np.array(logits_mlx.astype(mx.float32))
    V = len(logits)

    # 2. Stable softmax
    logits -= np.max(logits)
    np.exp(logits, out=logits)
    probs = logits / np.sum(logits)

    # 3. Reserve baseline of 1 count for every token in vocabulary
    # Remaining budget to distribute proportionally to probs:
    budget = M - V

    # Allocate budget
    scaled = (probs * budget).astype(np.int64)
    freqs = scaled + 1  # Strictly guaranteed freqs[i] >= 1

    # Adjust any rounding discrepancy on the argmax symbol
    current_sum = int(np.sum(freqs))
    diff = M - current_sum
    freqs[np.argmax(freqs)] += diff

    # 4. Cumulative sum table
    cum_freq = np.zeros(V + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum_freq[1:])

    return cum_freq, freqs


class RANSCoder:
    # Lower bound must satisfy L >= (2^32) for 32-bit streaming word emission
    L = 1 << 32

    @staticmethod
    def encode_tokens(
            tokens: list[int], cdf_slices: list[tuple[int, int]]
    ) -> bytes:
        state = RANSCoder.L
        words = []

        # Process symbols in reverse (LIFO order)
        for token, (start, freq) in reversed(list(zip(tokens, cdf_slices))):
            # Renormalization threshold: state < (L >> SCALE_BITS) * freq * 2^32
            # Emit 32-bit words if state is too large
            x_max = ((RANSCoder.L >> SCALE_BITS) * freq) << 32
            while state >= x_max:
                words.append(state & 0xFFFFFFFF)
                state >>= 32

            # C_rANS step
            state = ((state // freq) << SCALE_BITS) + (state % freq) + start

        # Flush 64-bit state
        stream_bytes = bytearray()
        stream_bytes.extend(struct.pack("<Q", state))

        # Reverse word order so forward decoder consumes sequentially
        for w in reversed(words):
            stream_bytes.extend(struct.pack("<I", w))

        return bytes(stream_bytes)

    @staticmethod
    def init_decoder(stream: bytes) -> tuple[int, list[int]]:
        state = struct.unpack("<Q", stream[:8])[0]
        words = []
        for i in range(8, len(stream), 4):
            words.append(struct.unpack("<I", stream[i: i + 4])[0])
        return state, words

    @staticmethod
    def decode_step(state: int) -> int:
        return state & (M - 1)

    @staticmethod
    def advance_decoder(
            state: int, start: int, freq: int, words: list[int], word_idx: int
    ) -> tuple[int, int]:
        # Invert C_rANS: D(x)
        state = freq * (state >> SCALE_BITS) + (state & (M - 1)) - start

        # Renormalize: pull 32-bit words while state < L
        while state < RANSCoder.L and word_idx < len(words):
            state = (state << 32) | words[word_idx]
            word_idx += 1

        return state, word_idx


# =====================================================================
# 3. Model Wrapper
# =====================================================================


class MLXRANSCoder(MLXCoder):

    def __init__(self, model_id: str = "mlx-community/LFM2.5-1.2B-Instruct-4bit"):
        super().__init__(model_id)

    def _next_logits(self, token_id: int, cache) -> mx.array:
        inp = mx.array([[token_id]])
        out = self.model(inp, cache=cache)
        mx.eval(out)
        return out[0, -1]

    def compress(self, text: str) -> bytes:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return b""

        total_tokens = len(tokens)
        print(f"Forward pass over {total_tokens} tokens...")

        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id
        cdf_slices = []

        for i, target_token in enumerate(tokens):
            logits = self._next_logits(current_token, cache)
            cum_freq, freqs = fast_logits_to_tables(logits)

            start = int(cum_freq[target_token])
            freq = int(freqs[target_token])
            cdf_slices.append((start, freq))

            current_token = target_token
            if (i + 1) % 50 == 0 or (i + 1) == total_tokens:
                print(f"  Token {i + 1}/{total_tokens}", end="\r")
        print()

        print("Executing backward rANS pass...")
        payload = RANSCoder.encode_tokens(tokens, cdf_slices)
        header = struct.pack(">I", total_tokens)
        return header + payload

    def decompress(self, payload: bytes) -> str:
        if len(payload) < 4:
            return ""

        (num_tokens,) = struct.unpack(">I", payload[:4])
        bitstream = payload[4:]

        print(f"Forward rANS decompression for {num_tokens} tokens...")
        state, words = RANSCoder.init_decoder(bitstream)
        word_idx = 0

        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id
        decoded_tokens = []

        for i in range(num_tokens):
            logits = self._next_logits(current_token, cache)
            cum_freq, freqs = fast_logits_to_tables(logits)

            slot = RANSCoder.decode_step(state)
            token_id = int(np.searchsorted(cum_freq, slot, side="right") - 1)

            start = int(cum_freq[token_id])
            freq = int(freqs[token_id])

            state, word_idx = RANSCoder.advance_decoder(
                state, start, freq, words, word_idx
            )

            decoded_tokens.append(token_id)
            current_token = token_id

            if (i + 1) % 50 == 0 or (i + 1) == num_tokens:
                print(f"  Decoded {i + 1}/{num_tokens}", end="\r")
        print()

        return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)


# Further optimisation: enable escapes
# =====================================================================
# 1. Escape-enabled rANS Coder
# =====================================================================


class EscapeRANSCoder:
    """64-bit streaming rANS encoder/decoder.

    M_BITS = 14 (16,384 counts). Fits comfortably in registers and ensures
    sufficient resolution for Top-K allocations.
    """

    M_BITS = 14
    M = 1 << M_BITS
    L = 1 << 31

    @staticmethod
    def encode_sequence(
            steps: list[tuple[int, int, int | None, int]],
    ) -> bytes:
        """Encodes a sequence of symbols in reverse (LIFO) order.

        Each step in `steps` is: (start_freq, symbol_freq, raw_literal_id,
        raw_bits). If raw_literal_id is not None, the literal is encoded directly
        onto the bitstream after the escape symbol.
        """
        state = EscapeRANSCoder.L
        words = []

        # Step backwards through the generated events
        for start, freq, raw_val, raw_bits in reversed(steps):
            # If an escape occurred, encode raw literal integer into the state first
            if raw_val is not None:
                # Range is 1 << raw_bits
                range_bits = raw_bits
                # Flush words if pushing raw integer exceeds state capacity
                while state >= (EscapeRANSCoder.L >> range_bits) << 32:
                    words.append(state & 0xFFFFFFFF)
                    state >>= 32
                state = (state << range_bits) | raw_val

            # Standard C_rANS state update
            x_max = ((EscapeRANSCoder.L >> EscapeRANSCoder.M_BITS) * freq) << 32
            while state >= x_max:
                words.append(state & 0xFFFFFFFF)
                state >>= 32

            state = (
                    ((state // freq) << EscapeRANSCoder.M_BITS) + (state % freq) + start
            )

        stream_bytes = bytearray()
        stream_bytes.extend(struct.pack("<Q", state))
        for w in reversed(words):
            stream_bytes.extend(struct.pack("<I", w))

        return bytes(stream_bytes)

    @staticmethod
    def init_decoder(stream: bytes) -> tuple[int, list[int]]:
        state = struct.unpack("<Q", stream[:8])[0]
        words = []
        for i in range(8, len(stream), 4):
            words.append(struct.unpack("<I", stream[i: i + 4])[0])
        return state, words

    @staticmethod
    def decode_slot(state: int) -> int:
        return state & (EscapeRANSCoder.M - 1)

    @staticmethod
    def advance_symbol(
            state: int, start: int, freq: int, words: list[int], word_idx: int
    ) -> tuple[int, int]:
        # Invert C_rANS
        state = (
                freq * (state >> EscapeRANSCoder.M_BITS)
                + (state & (EscapeRANSCoder.M - 1))
                - start
        )
        while state < EscapeRANSCoder.L and word_idx < len(words):
            state = (state << 32) | words[word_idx]
            word_idx += 1
        return state, word_idx

    @staticmethod
    def decode_raw_literal(
            state: int, raw_bits: int, words: list[int], word_idx: int
    ) -> tuple[int, int, int]:
        """Extracts raw integer of width `raw_bits` from the rANS stream."""
        val = state & ((1 << raw_bits) - 1)
        state >>= raw_bits
        while state < EscapeRANSCoder.L and word_idx < len(words):
            state = (state << 32) | words[word_idx]
            word_idx += 1
        return val, state, word_idx


# =====================================================================
# 2. Adaptive Top-K Distribution Builder
# =====================================================================


def build_adaptive_distribution(
        logits_mlx: mx.array, top_k: int = 1024, m_bits: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Constructs a localized CDF table for the Top-K tokens plus an ESCAPE symbol.

    Returns:
        (top_token_ids, cum_freq, freqs, escape_idx)
    """
    logits = np.array(logits_mlx.astype(mx.float32))
    V = len(logits)
    K = min(top_k, V - 1)

    # Stable softmax
    logits -= np.max(logits)
    np.exp(logits, out=logits)
    probs = logits / np.sum(logits)

    # Find Top-K indices efficiently
    top_token_ids = np.argpartition(-probs, K)[:K]
    # Sort top-k so ranking is deterministic
    top_token_ids = top_token_ids[np.argsort(-probs[top_token_ids])]

    top_probs = probs[top_token_ids]
    tail_prob = max(1.0 - float(np.sum(top_probs)), 1e-9)

    M = 1 << m_bits

    # Allocate frequency counts to Top-K + 1 (the Escape symbol at index K)
    # Ensure every symbol gets at least 1 count
    scaled_top = (top_probs * (M - (K + 1))).astype(np.int64) + 1
    escape_freq = max(int(round(tail_prob * (M - (K + 1)))), 1)

    freqs = np.append(scaled_top, escape_freq)

    # Adjust rounding to strictly sum to M
    current_sum = int(np.sum(freqs))
    diff = M - current_sum
    freqs[0] += diff  # Put any residual mass onto the #1 top token

    cum_freq = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=cum_freq[1:])

    escape_idx = K
    return top_token_ids, cum_freq, freqs, escape_idx


# =====================================================================
# 3. High-Level MLX Compressor with Escapes
# =====================================================================


class MLXEscapeRANSCoder:

    def __init__(
            self,
            model_id: str = "mlx-community/LFM2.5-1.2B-Instruct-4bit",
            top_k: int = 1024,
    ):
        print(f"Loading {model_id}...")
        self.model, self.tokenizer = load(model_id)
        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None) or 1
        self.top_k = top_k
        self.vocab_size = self.tokenizer.vocab_size
        # Number of bits needed to store raw literal token ID
        self.raw_bits = int(np.ceil(np.log2(self.vocab_size)))

    def _next_logits(self, token_id: int, cache) -> mx.array:
        inp = mx.array([[token_id]])
        out = self.model(inp, cache=cache)
        mx.eval(out)
        return out[0, -1]

    def compress(self, text: str) -> bytes:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return b""

        total_tokens = len(tokens)
        print(
            f"Collecting distributions (Top-{self.top_k} + Escape) for"
            f" {total_tokens} tokens..."
        )

        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id

        # List of (start_freq, freq, raw_literal_or_None, raw_bits)
        encoding_steps = []
        escapes_triggered = 0

        for i, target_token in enumerate(tokens):
            logits = self._next_logits(current_token, cache)
            top_ids, cum_freq, freqs, escape_idx = build_adaptive_distribution(
                logits, top_k=self.top_k, m_bits=EscapeRANSCoder.M_BITS
            )

            # Check if target token is within Top-K
            match = np.where(top_ids == target_token)[0]
            if len(match) > 0:
                idx = int(match[0])
                encoding_steps.append((cum_freq[idx], freqs[idx], None, 0))
            else:
                # Out-of-vocabulary for top-k -> Encode ESCAPE + Raw Token ID
                escapes_triggered += 1
                encoding_steps.append((
                    cum_freq[escape_idx],
                    freqs[escape_idx],
                    target_token,
                    self.raw_bits,
                ))

            current_token = target_token
            if (i + 1) % 50 == 0 or (i + 1) == total_tokens:
                print(
                    f"  Step {i + 1}/{total_tokens} (Escapes:"
                    f" {escapes_triggered})",
                    end="\r",
                )
        print()

        print(
            f"Executing rANS backward pass... (Escape rate:"
            f" {escapes_triggered / total_tokens:.2%})"
        )
        payload = EscapeRANSCoder.encode_sequence(encoding_steps)
        header = struct.pack(">II", total_tokens, self.top_k)
        return header + payload

    def decompress(self, payload: bytes) -> str:
        if len(payload) < 8:
            return ""

        num_tokens, top_k = struct.unpack(">II", payload[:8])
        bitstream = payload[8:]

        print(f"Decompressing {num_tokens} tokens with Forward Escape-rANS...")
        state, words = EscapeRANSCoder.init_decoder(bitstream)
        word_idx = 0

        cache = make_prompt_cache(self.model)
        current_token = self.bos_token_id
        decoded_tokens = []

        for i in range(num_tokens):
            logits = self._next_logits(current_token, cache)
            top_ids, cum_freq, freqs, escape_idx = build_adaptive_distribution(
                logits, top_k=top_k, m_bits=EscapeRANSCoder.M_BITS
            )

            slot = EscapeRANSCoder.decode_slot(state)
            symbol_idx = int(np.searchsorted(cum_freq, slot, side="right") - 1)

            start = int(cum_freq[symbol_idx])
            freq = int(freqs[symbol_idx])
            state, word_idx = EscapeRANSCoder.advance_symbol(
                state, start, freq, words, word_idx
            )

            if symbol_idx == escape_idx:
                # Consume raw literal token
                token_id, state, word_idx = EscapeRANSCoder.decode_raw_literal(
                    state, self.raw_bits, words, word_idx
                )
            else:
                token_id = int(top_ids[symbol_idx])

            decoded_tokens.append(token_id)
            current_token = token_id

            if (i + 1) % 50 == 0 or (i + 1) == num_tokens:
                print(f"  Decoded {i + 1}/{num_tokens}", end="\r")
        print()

        return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)

    def close(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if mx.metal.is_available():
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =====================================================================
# 3. Demonstration & Round-Trip Verification
# =====================================================================

if __name__ == "__main__":
    # Lightweight, fast model for testing
    # model_name = 'mlx-community/LFM2.5-1.2B-Instruct-4bit'
    model_name = 'mlx-community/Qwen3-0.6B-4bit'
    # model_name = 'rapid-mlx/Ling-3.0-tiny-MLX-4bit' # ValueError: Model type bailing_hybrid not supported.
    # model_name = "mlx-community/Qwen3.5-4B-MLX-4bit"
    # compressor = MLXArithmeticCoder("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    # compressor = MLXArithmeticCoder(model_name)
    # compressor = MLXRANSCoder(model_name)
    compressor = MLXEscapeRANSCoder(model_name)
    print_memory_usage()

    test_text = (
        "Entropy coding replaces next-token sampling with deterministic interval contraction. "
        "Because decompression mirrors the exact probability brackets produced by the autoregressive model, "
        "the recovered text is identical to the original input. Autoregressive neural networks make "
        "exceptionally strong compressors because their world model assigns high probabilities to "
        "plausible syntactic and semantic structures, directly minimizing cross-entropy loss."
    )
    with open(__file__, "r", encoding="utf-8") as f:
        test_text = f.read()

    print("\n--- Starting Compression ---")
    start_time_compress = time.time()
    raw_bytes = test_text.encode("utf-8")
    compressed = compressor.compress(test_text)
    end_time_compress = time.time()
    print(f"Original Text Length   : {len(raw_bytes)} bytes")
    print(f"Compressed Size (Total): {len(compressed)} bytes (incl. 4-byte header)")
    print(f"Compression Ratio      : {len(compressed) / len(raw_bytes):.2%}")
    print(f"Time Taken to Compress : {end_time_compress - start_time_compress:.2f} seconds")
    print("\n--- Starting Decompression ---")

    start_time_decompress = time.time()
    restored_text = compressor.decompress(compressed)
    end_time_decompress = time.time()
    print(f"Time Taken to Decompress: {end_time_decompress - start_time_decompress:.2f} seconds")

    # Clean up and reclaim VRAM/unified memory
    compressor.close()
    del compressor

    print_memory_usage()

    print(f"Matches Original: {restored_text == test_text}")
    if restored_text != test_text:
        print("\nOriginal Text:\n", test_text)
        print("\nDecoded Output:\n", restored_text)

    # Run side-by-side benchmark
    benchmark_compression(test_text, compressed)
