# Generalization check (llm-20261122-183752)

## loan: 100k new applicants (positives 31%)

| method | accuracy | AUROC | runs |
|---|---|---|---|
| always guess the common answer | 68.8% | 0.500 | – |
| trained simple net, 250 examples | 73.9% | 0.771 | 10 |
| trained simple net, all 1400 examples | 74.8% | 0.788 | 1 |
| trained small net, 250 examples | 80.8% | 0.844 | 10 |
| trained small net, all 1400 examples | 90.3% | 0.921 | 1 |
| **LLM weights, simple net (0 examples)** | 72.5% (71.1–74.6) | 0.787 | 4 |
| **LLM weights, small net (0 examples)** | 94.8% (94.6–94.9) | 0.932 | 4 |
| **LLM Python rule (0 examples)** | 94.9% (94.9–94.9) | 0.932 | 3 |

## loan: 100k SHIFTED population (positives 26%)

| method | accuracy | AUROC | runs |
|---|---|---|---|
| always guess the common answer | 74.4% | 0.500 | – |
| trained simple net, 250 examples | 72.8% | 0.756 | 10 |
| trained simple net, all 1400 examples | 73.6% | 0.765 | 1 |
| trained small net, 250 examples | 79.3% | 0.799 | 10 |
| trained small net, all 1400 examples | 90.0% | 0.894 | 1 |
| **LLM weights, simple net (0 examples)** | 69.0% (67.0–71.3) | 0.770 | 4 |
| **LLM weights, small net (0 examples)** | 94.9% (94.8–95.0) | 0.917 | 4 |
| **LLM Python rule (0 examples)** | 95.0% (95.0–95.0) | 0.916 | 3 |

## tumor: all 569 real cases (positives 37%)

| method | accuracy | AUROC | runs |
|---|---|---|---|
| always guess the common answer | 62.7% | 0.500 | – |
| trained simple net, 250 examples | 98.1% | 0.996 | 10 |
| trained simple net, all 398 (includes some of these cases) examples | 98.6% | 0.997 | 1 |
| trained small net, 250 examples | 98.3% | 0.997 | 10 |
| trained small net, all 398 (includes some of these cases) examples | 98.9% | 0.999 | 1 |
| **LLM weights, simple net (0 examples)** | 97.5% (97.0–98.2) | 0.995 | 4 |
| **LLM weights, small net (0 examples)** | 96.6% (95.3–97.4) | 0.992 | 4 |
| **LLM Python rule (0 examples)** | 96.4% (95.3–97.2) | 0.991 | 4 |
