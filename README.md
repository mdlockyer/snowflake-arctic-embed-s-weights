# snowflake-arctic-embed-s-weights

Deterministic pure-Swift SmoothQuant (`alpha = 0.40`) Int8 weight converter, `.ae8` pack builder,
and FP32/FP64 reference test vectors for [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s).

This repository produces the shared, CPU-agnostic `model.ae8` pack and reference test fixtures consumed by:
- [`snowflake-arctic-embed-s-swift`](https://github.com/mdlockyer/snowflake-arctic-embed-s-swift)
- [`snowflake-arctic-embed-s-rust`](https://github.com/mdlockyer/snowflake-arctic-embed-s-rust)

## Artifacts (`v1.0.0`)

| Asset | SHA-256 | Description |
|---|---|---|
| `model.ae8` | `d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1` | SmoothQuant W8A8 Int8 weights (33.8 MB) |
| `vocab.txt` | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` | Unmodified BERT WordPiece vocabulary |
| `arctic-embed-s-ae8.tar.gz` | See `SHA256SUMS` | Complete bundle (`model.ae8`, `vocab.txt`, `test-data/`, `LICENSE`, `NOTICE`, `SHA256SUMS`) |

See [`FORMAT.md`](FORMAT.md) for the binary layout specification of `model.ae8`.

## Building the Pack Locally

```bash
scripts/download_weights.sh     # downloads HF revision e596f507 into ./weights (SHA-256 verified)
scripts/build_release_pack.sh   # runs arctic-embed-convert, verifies SHA-256, and writes dist/
```

Or run the converter directly:

```bash
swift run -c release arctic-embed-convert weights/model.safetensors dist/model.ae8
```

## Testing

```bash
swift test -c release
```

- `testDetMathMatchesLibmClosely`: verifies the platform-independent IEEE-754 `DetMath` routines (`expF64`, `lnF64`, `pow`, `gelu`).
- `testSafetensorsConversionDeterminismAndRoundtrip`: converts `weights/model.safetensors` to `.ae8`, verifies its SHA-256 matches `d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1`, and verifies `.ae8` round-trip serialization.

## Provenance & License

- Source checkpoint: [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) revision `e596f507467533e48a2e17c007f0e1dacc837b33` (`model.safetensors` SHA-256 `0350e62666ee2403db0223fc7cef6293c951cc63d1d2e71468da9a5aea3b544a`).
- Licensed under the [Apache License, Version 2.0](LICENSE); see [`NOTICE`](NOTICE).
