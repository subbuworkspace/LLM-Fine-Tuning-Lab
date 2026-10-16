# Fine-Tuning Concepts — Quick Reference

## What fine-tuning does
Fine-tuning continues training a pretrained model on curated examples so its behavior better matches a task, style, format, or domain. It does not automatically make facts current or guarantee truth.

## Core vocabulary
- **Pretraining:** learning broad patterns from a large corpus.
- **SFT (Supervised Fine-Tuning):** training on input/output demonstrations.
- **Instruction tuning:** SFT examples phrased as instructions; a common form of SFT.
- **Full fine-tuning:** update most/all model weights; costly and can cause forgetting.
- **PEFT:** parameter-efficient fine-tuning; update a small set of parameters.
- **LoRA:** add trainable low-rank matrices to selected model layers while freezing base weights.
- **QLoRA:** keep the base model quantized (often 4-bit) and train LoRA adapters.
- **Quantization:** represent weights with fewer bits to reduce memory; may trade precision for efficiency.
- **Adapter:** small learned parameter set that can be stored separately from the base model.
- **DPO:** Direct Preference Optimization; train from preferred vs rejected responses without a separate online RL loop.
- **RLHF:** use human preferences, often via a reward model and reinforcement learning, to shape behavior.
- **GRPO/PPO:** reinforcement-learning optimization methods; advanced topics, not required for the starter project.
- **RAG:** retrieve external information at inference time and provide it as context.
- **Distillation:** train a smaller student model to imitate a teacher model.
- **Catastrophic forgetting:** tuned model loses some capabilities it had before.
- **Data leakage:** evaluation information accidentally appears in training data or preprocessing.
- **Perplexity/loss:** token prediction metrics; useful but not sufficient to prove task usefulness.

## Choose the right approach
- Try **prompting** first when instructions can reliably steer the base model.
- Use **RAG** when answers depend on private, changing, or source-citable knowledge.
- Consider **fine-tuning** when you need consistent task behavior, domain-specific patterns, response structure, or style and have representative examples.
- Consider **distillation** when latency/cost/size is the key constraint.
- Often the best solution combines RAG and fine-tuning.

## Typical workflow
1. Define success criteria and a baseline.
2. Create licensed, privacy-safe, representative examples.
3. Deduplicate and normalize; remove secrets and low-quality labels.
4. Split train/validation/test before tuning; avoid near-duplicate leakage.
5. Select a compatible base model and check its license.
6. Start with a small SFT/LoRA experiment.
7. Track configuration, seeds, loss, runtime, GPU memory and adapter version.
8. Compare base vs tuned model on the same held-out cases.
9. Test robustness, safety, formatting and regression on unrelated tasks.
10. Document limitations; monitor and roll back if quality degrades.

## LoRA intuition
For a frozen weight matrix W, LoRA learns a low-rank update ΔW = B·A. The effective weight is approximately W + (alpha/r)·B·A. Only A and B are trained, so far fewer parameters change. Rank `r` controls adapter capacity; alpha scales the update; dropout can regularize. Exact useful settings depend on model, task and data.

## QLoRA intuition
QLoRA combines a quantized frozen base model with trainable LoRA adapters. NF4, double quantization and careful compute dtype are common memory-saving choices. GPU support and library versions matter; not every environment supports bitsandbytes equally well.

## Evaluation checklist
- Task metric: exact match, accuracy, F1, structured-output validity, or domain rubric.
- Human review for correctness, clarity, bias and harmful behavior.
- Baseline comparison using identical prompts and held-out examples.
- Slice metrics by category, language, difficulty and edge cases.
- Regression/safety tests and confidence intervals where feasible.
- Do not report training loss alone as proof of real-world improvement.
