import fire, os
import torch
import datasets
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
from unsloth import FastLanguageModel
import wandb

def load_model(model_name_or_path, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        padding_side='right'
    )
    
    local_rank = 0
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=None,
        device_map=f"cuda:{local_rank}",
        load_in_4bit=True,
        use_cache=False,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=256, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=512,
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias="none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ,
        modules_to_save= ["embed_tokens", "lm_head"],
    )
        
    return tokenizer, model

def load_dataset(tokenizer: AutoTokenizer, dataset_name_or_path: str, max_seq_length: int):

    def put_in_list(examples):
        result = {"conversation": []}
        for output, instruction, input in zip(examples["output"],examples["instruction"],examples["input"]):
            dict_ = [{"output": output, "instruction" : instruction, "input": input}]
            result["conversation"].append(dict_)
        return result
    
    def convert_conversation_to_input(examples):
        input_ids = [tokenizer.apply_chat_template(conversation, tokenize=True)[: max_seq_length] for conversation in examples["conversation"]]
        return {'input_ids': input_ids}
    
    dataset = datasets.load_dataset(dataset_name_or_path, split='train[:]')
    dataset = dataset.map(put_in_list, batched=True)
    dataset = dataset.map(convert_conversation_to_input, remove_columns=list(dataset.features), batched=True)
    
    dataset = dataset.train_test_split(test_size=1000, shuffle=True)
    return dataset['train'], dataset['test']

def train(
    model_name_or_path: str = "/home/user/LLAMA3_for_coding/finetuning/Meta-Llama-3-8b",
    train_batch_size: int = 8,
    max_seq_length: int = 8192,
):
    dataset_name = 'dinhlnd1610/CodeAlpaca-AddLanguage'
    output_dir = "unsloth-lora-checkpoints-r256a512-31_5_11am"
    
    # Load model and tokenizer
    tokenizer, model = load_model(model_name_or_path, max_seq_length)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset(tokenizer=tokenizer, dataset_name_or_path=dataset_name, max_seq_length=max_seq_length)
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|begin_of_text|>",
        response_template="<|start_header_id|><|output|><|end_header_id|>\n\n",
        tokenizer=tokenizer,
        mlm=False,
)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        bf16=True,
        learning_rate=2e-4,
        optim="adamw_8bit",
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy = "epoch",
        save_total_limit=1,
        num_train_epochs=1,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name="llama3_new",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_dir)
    wandb.finish()

if __name__ == "__main__":
    # fire.Fire(train)
    fire.Fire(train)