# LLM Fine-Tuning Interview Questions and Short Answers

## Fundamentals
1. **What is fine-tuning?** Continuing training of a pretrained model on task-specific examples to change its behavior.
2. **Pretraining vs fine-tuning?** Pretraining learns broad representations from huge corpora; fine-tuning adapts an existing model to a narrower objective.
3. **What is SFT?** Supervised learning on demonstrations of desired inputs and outputs.
4. **Instruction tuning vs SFT?** Instruction tuning is SFT where examples are designed to teach instruction following.
5. **When would you not fine-tune?** When prompting solves it, facts change frequently (use RAG), there is too little trustworthy data, or cost/risk exceeds expected benefit.
6. **Fine-tuning vs RAG?** Fine-tuning changes model parameters/behavior; RAG supplies retrieved context at inference. RAG is usually better for changing facts and citations.
7. **Can fine-tuning reliably add fresh facts?** It can memorize examples, but it is not a dependable database update mechanism; use retrieval for freshness.

## LoRA / QLoRA / PEFT
8. **What is PEFT?** Parameter-efficient methods that train only a small subset of parameters.
9. **Explain LoRA.** It freezes original weights and learns a low-rank update, reducing trainable parameters and optimizer memory.
10. **What does LoRA rank mean?** The rank controls the capacity/size of the low-rank update; larger ranks cost more and are not always better.
11. **What are alpha and dropout?** Alpha scales the LoRA update; dropout may regularize adapter training.
12. **What is QLoRA?** LoRA training with a quantized frozen base model, often 4-bit, to reduce memory use.
13. **Does 4-bit mean the whole training process uses 4-bit?** No. The base weights are quantized; adapter parameters and computation can use other precisions.
14. **What are target modules?** Model layers receiving adapters, commonly attention projections and sometimes other linear layers.
15. **Can you merge an adapter?** Often yes, depending on model/adapter/quantization support; test numerical and deployment behavior.

## Data and training
16. **What makes a good dataset?** Correct, diverse, representative, licensed, privacy-safe, consistently labeled and aligned to production use.
17. **How do you avoid leakage?** Split before training, deduplicate, group related records, and keep test examples out of prompt construction and tuning.
18. **Why use the tokenizer's chat template?** Models expect specific role markers and boundaries; wrong formatting can degrade learning.
19. **What is overfitting?** Training performance improves while held-out performance stops improving or worsens.
20. **How do you reduce overfitting?** Better/more data, fewer epochs, lower capacity, regularization, early stopping and careful validation.
21. **What is catastrophic forgetting?** New training damages prior capabilities; mitigate with representative mixed data, conservative settings and regression tests.
22. **What are gradient accumulation and batch size?** Accumulation combines gradients over several mini-batches to approximate a larger effective batch.
23. **What does learning rate do?** It controls update size; excessive values destabilize training, while tiny values may learn too slowly.
24. **Why save checkpoints?** Recovery, comparison, reproducibility and rollback.

## Evaluation and production
25. **Why isn't loss enough?** Lower token loss does not guarantee correct task outputs, safety or usefulness.
26. **How do you evaluate?** Compare base and tuned models on the same held-out set using task metrics plus human, robustness and safety review.
27. **What is data contamination?** Evaluation examples or close variants leak into training, inflating reported results.
28. **How do you evaluate classification?** Accuracy plus per-class precision, recall, F1 and confusion matrix, especially with imbalance.
29. **How would you deploy an adapter?** Load the compatible base model plus adapter, pin versions, benchmark latency/memory and monitor quality.
30. **What should a model card include?** Intended use, base model, data provenance, training config, evaluation, limitations, license and safety considerations.
31. **How do you handle private data?** Minimize, redact, obtain authorization, follow retention/access rules and never upload confidential records to public services.
32. **How do you estimate GPU memory?** Account for weights, activations, gradients, optimizer states, sequence length, batch size and framework overhead; benchmark actual hardware.

## Preference tuning / advanced methods
33. **What is DPO?** A preference-optimization method using chosen/rejected response pairs, typically avoiding a separate reward-model training loop.
34. **RLHF vs DPO?** RLHF commonly trains a reward model and uses RL; DPO optimizes preference pairs directly with a simpler pipeline.
35. **What is a reward model?** A model trained to score outputs according to preferences or a target reward.
36. **What are PPO and GRPO?** Reinforcement-learning optimization methods used in some post-training pipelines; know the high-level trade-offs before diving into implementation.
37. **What is distillation?** Training a smaller model to reproduce a stronger teacher's behavior.
38. **How do you choose between LoRA and full fine-tuning?** Compare task complexity, compute, data scale, desired capacity, deployment constraints and measured results.

## Scenario questions
39. **A tuned model has low loss but worse accuracy. Why?** Data/metric mismatch, leakage, overfitting, bad chat template, label noise or decoding differences; inspect error slices and baseline.
40. **A model returns extra text instead of a label. What do you do?** Improve examples/template, validate output constraints, evaluate exact match, and consider constrained decoding where supported.
41. **Your dataset is only 100 examples. What next?** Start with prompting/baseline, audit labels, use carefully reviewed augmentation if appropriate, use PEFT conservatively and report uncertainty.
42. **The knowledge changes daily. Fine-tune or RAG?** Usually RAG, because updates can happen without retraining.
43. **Training runs out of memory. What can you change?** Reduce sequence length/batch size, use gradient accumulation, checkpointing, quantization/PEFT, or a smaller model.
44. **How do you prove value to a business?** Define task KPIs, compare against baseline, include error cost, latency and serving cost, and test on representative held-out production-like data.
45. **The model becomes less safe after tuning. What do you do?** Stop rollout, evaluate failure modes, improve data and safeguards, retrain/test and retain rollback capability.

## Strong answer structure
Use: **problem → alternatives → data → method → baseline → evaluation → trade-offs → deployment/monitoring**. Interviewers value justified decisions and measured outcomes more than memorized library syntax.
