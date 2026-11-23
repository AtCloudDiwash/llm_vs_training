# Can an LLM replace training data? (pilot-20261122-183428)

## tumor

| method | accuracy | AUROC | worth about N examples | valid runs |
|---|---|---|---|---|
| training, simple net, all 8 sizes | n=5: 89.6%, n=10: 88.5%, n=25: 92.5%, n=50: 93.8%, n=100: 95.2%, n=250: 97.7% | | | 20 each |
| training, small net, all 8 sizes | n=5: 87.2%, n=10: 86.8%, n=25: 91.8%, n=50: 93.8%, n=100: 94.8%, n=250: 97.0% | | | 20 each |
| random weights | 52.9% | 0.540 | | 20 |
| LLM weights, simple net | 97.1% (97.1–97.1) | 0.998 | about 212 | 1/1 |
| LLM weights, small net | – | – | – | 0/0 |
| LLM Python rule | – | – | – | 0/0 |
| LLM weights, names hidden | – | – | – | 0/0 |

## loan

| method | accuracy | AUROC | worth about N examples | valid runs |
|---|---|---|---|---|
| training, simple net, all 8 sizes | n=5: 60.3%, n=10: 60.1%, n=25: 64.1%, n=50: 68.2%, n=100: 70.6%, n=250: 71.8% | | | 20 each |
| training, small net, all 8 sizes | n=5: 60.5%, n=10: 60.9%, n=25: 65.3%, n=50: 70.3%, n=100: 74.1%, n=250: 79.4% | | | 20 each |
| random weights | 45.3% | 0.432 | | 20 |
| LLM weights, simple net | – | – | – | 0/0 |
| LLM weights, small net | – | – | – | 0/0 |
| LLM Python rule | – | – | – | 0/0 |
| LLM weights, names hidden | – | – | – | 0/0 |
