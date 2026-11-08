# `.ae8` Binary Model Format Specification

The `.ae8` file (`model.ae8`, 33.8 MB) is a CPU-agnostic, little-endian binary serialization of the
SmoothQuant (`alpha = 0.40`) Int8-quantized weights of `Snowflake/snowflake-arctic-embed-s`.

Both [`snowflake-arctic-embed-s-swift`](https://github.com/mdlockyer/snowflake-arctic-embed-s-swift) and
[`snowflake-arctic-embed-s-rust`](https://github.com/mdlockyer/snowflake-arctic-embed-s-rust) read this exact
binary format and swizzle the row-major weight matrices into their host CPU's SIMD tile layout at startup.

## Primitives

- `u8`: 1-byte unsigned integer
- `u64`: 8-byte unsigned little-endian integer
- `i8_vec`: `u64` element count `N`, followed by `N` contiguous `Int8` bytes
- `f32_vec`: `u64` element count `N`, followed by `N` contiguous IEEE-754 little-endian `Float32` values (`4 * N` bytes)
- `linear`:
  - `n_out: u64`
  - `k_in: u64`
  - `w_q: i8_vec` (`n_out * k_in` row-major quantized weights; `n_out == 0` when unused)
  - `w_scale: f32_vec` (`n_out` per-output-channel dequantization scales)
  - `bias: f32_vec` (`n_out` FP32 bias values)
  - `inv_smooth: f32_vec` (`k_in` per-input-channel `1.0 / smooth` multipliers applied before dynamic activation quantization)

## Layout

1. **Magic header (8 bytes):** `"ARCTIC8\x01"` (`[0x41, 0x52, 0x43, 0x54, 0x49, 0x43, 0x38, 0x01]`)
2. **Embeddings:**
   - `word_emb_q: i8_vec` (`30522 * 384`)
   - `word_emb_scale: f32_vec` (`30522`)
   - `pos_emb_q: i8_vec` (`512 * 384`)
   - `pos_emb_scale: f32_vec` (`512`)
   - `type_emb_q: i8_vec` (`2 * 384`)
   - `type_emb_scale: f32_vec` (`2`)
   - `emb_ln_weight: f32_vec` (`384`)
   - `emb_ln_bias: f32_vec` (`384`)
3. **12 Transformer Layers (`l = 0..11`):**
   - `qkv: linear` (`[1152, 384]` for layers `0..10`; empty `n_out = 0` carrier with `inv_smooth[384]` for layer `11`)
   - `has_split: u8` (`0` for layers `0..10`; `1` for layer `11` `[CLS]`-only pruning)
   - If `has_split == 1`:
     - `q_only: linear` (`[384, 384]`)
     - `kv_only: linear` (`[768, 384]`)
   - `attn_out: linear` (`[384, 384]`)
   - `ln1_weight: f32_vec` (`384`)
   - `ln1_bias: f32_vec` (`384`)
   - `ffn_inter: linear` (`[1536, 384]`)
   - `ffn_out: linear` (`[384, 1536]`)
   - `ln2_weight: f32_vec` (`384`)
   - `ln2_bias: f32_vec` (`384`)
