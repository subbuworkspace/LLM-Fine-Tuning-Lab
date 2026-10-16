# LLM Fine-Tuning Lab: SFT, LoRA & QLoRA

A beginner-friendly, portfolio-ready repository that demonstrates an end-to-end parameter-efficient fine-tuning workflow using Hugging Face Transformers, TRL and PEFT.

**Mini-project:** fine-tune a small instruction model to classify synthetic support tickets into `account_access`, `billing`, `bug_report`, or `feature_request`.

> Educational demo only. The included dataset is intentionally tiny and synthetic. It is not evidence of production performance. Use a GPU runtime and compare the tuned model against a baseline before making any claims.

## What you'll learn

- Pretraining vs fine-tuning, SFT and instruction tuning
- Full fine-tuning vs PEFT
- LoRA, QLoRA, quantization, adapters and target modules
- Dataset quality, chat templates, leakage and held-out evaluation
- Training configuration, checkpoints, inference and common failure modes
- When to choose prompting, RAG, fine-tuning or distillation
- Interview questions covering fundamentals through DPO/RLHF/GRPO

## Repository structure

```text
llm-fine-tuning-lab/
├── data/
│   ├── train.jsonl
│   └── eval.jsonl
├── train_qlora.py
├── predict.py
├── evaluate.py
├── CONCEPTS.md
├── INTERVIEW_QUESTIONS.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start (recommended: Google Colab GPU)

1. Create a new Colab notebook and choose **Runtime → Change runtime type → T4 GPU** (or another supported CUDA GPU).
2. Upload/clone this repository or copy the files into the Colab workspace.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify the runtime has a GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU not found")
```

5. Train the QLoRA adapter:

```bash
python train_qlora.py
```

6. Run a prediction after training:

```bash
python predict.py --ticket "I was billed twice for the same month."
```

7. Evaluate on the tiny held-out examples:

```bash
python evaluate.py
```

The scripts use `Qwen/Qwen2.5-0.5B-Instruct` as a compact starter model. Model access, license, runtime compatibility and library APIs can change; check the model card and current library docs before running. A GPU is required by the starter training script because it uses 4-bit bitsandbytes quantization.

## What the training code does

1. Loads JSONL instruction/response examples.
2. Formats each example with the model's chat template.
3. Loads the base model in 4-bit NF4 quantization.
4. Freezes the base model and attaches LoRA adapters to linear layers.
5. Trains the adapter with TRL's `SFTTrainer`.
6. Saves the adapter and tokenizer under `outputs/support-ticket-qlora/`.

The adapter is not a standalone full model: inference loads the compatible base model and the adapter.

## Key concepts in plain English

| Concept | Simple explanation |
|---|---|
| SFT | Show the model examples of desired inputs and outputs |
| PEFT | Train a small number of parameters instead of updating all model weights |
| LoRA | Learn a compact low-rank update to selected model layers |
| QLoRA | Quantize the frozen base weights and train LoRA adapters to reduce memory |
| RAG | Retrieve relevant documents at answer time; useful for changing or source-grounded facts |
| DPO | Learn from chosen/rejected response pairs |
| Catastrophic forgetting | Fine-tuning harms some previously learned capabilities |
| Data leakage | Test examples or close duplicates sneak into training |

Read [CONCEPTS.md](CONCEPTS.md) for the workflow and glossary, and [INTERVIEW_QUESTIONS.md](INTERVIEW_QUESTIONS.md) for interview preparation.

## Important evaluation note

The starter evaluation set has only four records, so its score is not statistically meaningful. Expand the dataset, remove duplicates, use a proper train/validation/test split, and compare the base model and fine-tuned model using the same held-out examples and decoding settings. Report per-class precision/recall/F1 and human-reviewed error analysis for a serious classification experiment.

Do not claim the model improved unless your measurements support that conclusion.

## Common issues

- **`No CUDA GPU detected`:** use a supported GPU runtime such as Colab with GPU enabled.
- **`bitsandbytes` install/runtime error:** use a compatible Linux/CUDA environment and check current bitsandbytes documentation; Windows support can differ.
- **TRL argument mismatch:** package APIs evolve. Create a clean virtual environment and check the installed TRL docs/version if `SFTConfig` or `SFTTrainer` arguments differ.
- **Out of memory:** reduce `max_length`, batch size or LoRA target modules; use gradient accumulation or a smaller model.
- **Poor predictions:** the dataset is tiny by design. Add diverse, reviewed examples, compare with a base-model baseline, and inspect errors rather than merely training for more epochs.

## Responsible use

- Use only data you are allowed to use.
- Never commit API keys, private records or confidential company information.
- Check the base model's license and usage terms.
- Document dataset provenance, intended use, limitations and evaluation.
- Fine-tuning does not guarantee factuality, fairness or safety.

## References

- Hugging Face TRL quickstart: https://huggingface.co/docs/trl/main/en/quickstart
- Hugging Face TRL PEFT integration: https://huggingface.co/docs/trl/main/en/peft_integration
- Hugging Face PEFT documentation: https://huggingface.co/docs/peft/index
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/index
- QLoRA paper: https://arxiv.org/abs/2305.14314
- LoRA paper: https://arxiv.org/abs/2106.09685

## Suggested next improvements

1. Add at least hundreds of reviewed examples and a validation split.
2. Build a base-vs-tuned evaluation table and confusion matrix.
3. Add a small Streamlit demo.
4. Track experiments with W&B or MLflow.
5. Publish the adapter only after checking license, data rights and evaluation quality.
6. Try a DPO exercise with preference pairs after mastering SFT.

## License

Choose a license for your own code after reviewing dependencies and the base model's terms. The base model and datasets retain their own licenses.
