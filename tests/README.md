# Unit tests (CPU)

Checks of the code itself, not the alignment tests of the paper. They run on CPU with synthetic scans and random
weights, need no data, GPU or download, and take about 15 seconds:

```bash
python -m pytest tests/        # or run each file: python tests/test_smoke_cpu.py
```

| File | Checks |
|---|---|
| `test_smoke_cpu.py` | organ edits, every encoder loss (finite, gradients flow), AUROC and matched-pair AUROC, the evaluation harness end to end with a toy encoder |
| `test_fixed_route.py` | organ routing: a sentence reads only the tokens of the organs it names |
| `test_native_prompts.py` | tumour findings use each released encoder's own prompt template |

To check a GPU setup with real data, run `bash scripts/setup/smoke_test.sh` (see the main README).
