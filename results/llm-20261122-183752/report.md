# Can an LLM replace training data? (llm-20261122-183752)

## tumor

| method | accuracy | AUROC | worth about N examples | valid runs |
|---|---|---|---|---|
| training, simple net | n=5: 89.6%, n=10: 88.5%, n=25: 92.5%, n=50: 93.8%, n=100: 95.2%, n=250: 97.7% | | | 20 each |
| training, small net | n=5: 87.2%, n=10: 86.8%, n=25: 91.8%, n=50: 93.8%, n=100: 94.8%, n=250: 97.0% | | | 20 each |
| random weights | 52.9% | 0.540 | | 20 |
| LLM weights, simple net | 97.1% (96.5–98.2) | 0.997 | about 213 | 4/5 |
| LLM weights, small net | 95.3% (93.6–96.5) | 0.992 | about 135 | 4/5 |
| LLM Python rule | 95.5% (94.2–96.5) | 0.992 | about 114 | 4/5 |
| LLM weights, names hidden | 63.7% (63.7–63.7) | 0.500 | fewer than 5 | 4/5 |

### tumor/simple: start from LLM weights vs from scratch, then train

| examples | from LLM | from scratch |
|---|---|---|
| 5 | 86.7% | 88.6% |
| 10 | 88.9% | 90.0% |
| 25 | 91.9% | 92.5% |
| 50 | 93.5% | 93.8% |

### tumor/small: start from LLM weights vs from scratch, then train

| examples | from LLM | from scratch |
|---|---|---|
| 5 | 84.9% | 85.6% |
| 10 | 89.0% | 89.4% |
| 25 | 91.6% | 91.7% |
| 50 | 93.7% | 93.9% |

## loan

| method | accuracy | AUROC | worth about N examples | valid runs |
|---|---|---|---|---|
| training, simple net | n=5: 60.3%, n=10: 60.1%, n=25: 64.1%, n=50: 68.2%, n=100: 70.6%, n=250: 71.8% | | | 20 each |
| training, small net | n=5: 60.5%, n=10: 60.9%, n=25: 65.3%, n=50: 70.3%, n=100: 74.1%, n=250: 79.4% | | | 20 each |
| random weights | 45.3% | 0.432 | | 20 |
| LLM weights, simple net | 71.4% (69.7–73.7) | 0.771 | about 200 | 4/5 |
| LLM weights, small net | 94.0% (93.8–94.2) | 0.919 | more than 250 | 4/5 |
| LLM Python rule | 94.2% (94.2–94.2) | 0.923 | more than 250 | 3/5 |
| LLM weights, names hidden | 65.8% (65.8–65.8) | 0.500 | about 36 | 3/5 |

### loan/simple: start from LLM weights vs from scratch, then train

| examples | from LLM | from scratch |
|---|---|---|
| 5 | 61.0% | 60.0% |
| 10 | 62.9% | 62.1% |
| 25 | 65.7% | 65.7% |
| 50 | 68.5% | 68.5% |

### loan/small: start from LLM weights vs from scratch, then train

| examples | from LLM | from scratch |
|---|---|---|
| 5 | 67.6% | 60.7% |
| 10 | 70.1% | 63.8% |
| 25 | 79.3% | 66.8% |
| 50 | 79.8% | 68.6% |
