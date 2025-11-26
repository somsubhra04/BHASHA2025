#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, argparse, torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# -----------------------
# Prompt & Data Utilities
# -----------------------
INSTR = "नीचे दिए गए वाक्य की व्याकरण-सुधारित रूप प्रदान करें।"
PROMPT_TEMPLATE = (
    "### निर्देश:\n"
    f"{INSTR}\n\n"
    "### इनपुट:\n{{src}}\n\n"
    "### उत्तर:\n"
)

def read_parallel(src_path: str, tgt_path: str):
    with open(src_path, "r", encoding="utf-8") as fsrc, open(tgt_path, "r", encoding="utf-8") as ftgt:
        src_lines = [l.strip() for l in fsrc.readlines()]
        tgt_lines = [l.strip() for l in ftgt.readlines()]
    assert len(src_lines) == len(tgt_lines), "src/tgt line count mismatch"
    return list(zip(src_lines, tgt_lines))

class GECDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=2048):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        prompt = PROMPT_TEMPLATE.format(src=src)
        full = prompt + tgt + self.tok.eos_token

        enc = self.tok(full, max_length=self.max_len, truncation=True)
        prompt_ids = self.tok(prompt, max_length=self.max_len, truncation=True)["input_ids"]

        labels = enc["input_ids"][:]
        mask_len = len(prompt_ids)
        labels[:mask_len] = [-100] * min(mask_len, len(labels))

        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lbl = f["labels"]
            pad_len = max_len - len(lbl)
            lbl = lbl + ([-100] * pad_len)
            labels.append(lbl)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

# -----------------------
# Training Pipeline
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="sarvamai/sarvam-m")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="outputs/sarvam_m_gec_lora")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_src = os.path.join(args.data_dir, "train", "train.src")
    train_tgt = os.path.join(args.data_dir, "train", "train.tgt")
    valid_src = os.path.join(args.data_dir, "valid", "valid.src")
    valid_tgt = os.path.join(args.data_dir, "valid", "valid.tgt")

    train_pairs = read_parallel(train_src, train_tgt)
    valid_pairs = read_parallel(valid_src, valid_tgt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    print("Loading base model in 4-bit quantized mode...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False

    # Attach LoRA adapters
    print("Attaching LoRA adapters...")
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        target_modules=lora_targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable params: {len(trainable)} / Total params: {sum(1 for _ in model.parameters())}")

    train_ds = GECDataset(train_pairs, tokenizer, max_len=args.max_len)
    valid_ds = GECDataset(valid_pairs, tokenizer, max_len=args.max_len)
    collator = DataCollatorForCausalLM(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=1,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit",
        report_to="none",
        gradient_checkpointing=True,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ----------- MERGE & SAVE -----------
    print("Merging LoRA and saving final model...")
    merged = trainer.model.merge_and_unload()
    merged.config.pad_token_id = tokenizer.pad_token_id

    # patch invalid generation_config
    if hasattr(merged, "generation_config"):
        gen_cfg = merged.generation_config
        if hasattr(gen_cfg, "do_sample") and gen_cfg.do_sample is False:
            if hasattr(gen_cfg, "temperature"):
                gen_cfg.temperature = None

    final_path = os.path.join(args.output_dir, "final_merged")
    os.makedirs(final_path, exist_ok=True)
    merged.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Final merged model saved at: {final_path}")

if __name__ == "__main__":
    main()
