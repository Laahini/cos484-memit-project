# Editing Memory in Transformers at Scale

### A Benchmark and Robustness Study of MEMIT

Final project for **COS 484: Natural Language Processing** (Princeton, Spring 2026).

MEMIT ([Meng et al.](https://arxiv.org/abs/2210.07229)) updates factual knowledge in a
language model by directly editing MLP weights across a band of mid-network layers,
scaling to thousands of simultaneous edits without retraining. It works well on
GPT-family models. This project asks two questions the original paper leaves open:

1. **Does it transfer?** Does MEMIT work on a modern non-GPT architecture (Llama 3.1)?
2. **Does it hold?** Do edits survive adversarial prompting, and do they stay contained
   to their own domain?

## Findings

**There is a substantial architecture gap.** MEMIT holds 100% of clean edits on
GPT-J 6B but only 62% on Llama 3.1 8B under the best configuration found
(layers 8–13, λ = 1000). MEMIT's MLP-as-key-value assumption does not cleanly
transfer beyond GPT-style models.

**Edits are shallower than they look.** On GPT-J, few-shot and persona attacks revert
more than two-thirds of edits — MEMIT shifts probability mass toward the injected
answer without fully erasing the original representation.

**Cross-domain interference is minimal on GPT-J.** Edits are precise and
non-interfering. On Llama the edit success rate is too low for the comparable
ablation to say anything reliable.

Together these position architectural specificity and adversarial robustness as
open challenges for scalable model editing.

## Notebooks

Written for Colab with GPU. Llama 3.1 8B is gated, so those notebooks authenticate
via `huggingface-cli login` or a Colab `HF_TOKEN` secret. All notebooks are committed
with their outputs intact.

| Notebook | What it does |
|---|---|
| `MEMIT_NLP.ipynb` | Baseline reproduction on GPT-J 6B; CounterFact evaluation |
| `MEMIT_Llama_Sweeps.ipynb` | Port to Llama 3.1 8B; layer-range, λ, and edit-count sweeps |
| `MEMIT_Jailbreak_GPTJ.ipynb` | Ablation (a) — do adversarial prompts revert edits? |
| `MEMIT_Jailbreak_Llama.ipynb` | Ablation (a) on Llama |
| `MEMIT_CrossDomain_GPTJ.ipynb` | Ablation (b) — do edits in one domain damage others? |
| `MEMIT_CrossDomain_Llama.ipynb` | Ablation (b) on Llama |

Suggested reading order: baseline reproduction → Llama port and sweeps → the two
ablations, GPT-J before Llama in each.

A note on scale: the CounterFact evaluation runs 100 edits rather than the paper's
10,000. That ceiling was compute, not design.

## Paper

- `paper/Final_Project_Paper.pdf` — compiled final paper
- `paper/final_paper.tex` — LaTeX source
- `paper/paper_extensions.tex` — extended write-up
- `paper/COS 484 Final Poster.pdf` — final poster
- `paper/Project Proposal.pdf` — original proposal

## Not included

The published MEMIT paper, course handouts, and example papers from other students
live alongside this work locally but are not redistributed here.

## Author

Laahini (`la0047`), COS 484, Spring 2026.
