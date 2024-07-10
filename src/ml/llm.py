# === DESCRIPTION ======================================================================================================
"""
LLM classes & methods
"""

# === DEPENDENCIES =====================================================================================================
import torch
import transformers
import peft
import unsloth
import datasets
from src.utils.path_utils import path
from pathlib import Path
from trl import SFTTrainer


# === CONSTANTS ========================================================================================================
MODEL_PATH = path("models/TODO")
MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 512
DATASET_MAPPING = {  # Mapping for dataset input
    "role": "from",
    "content": "value",
    "user": "human",
    "assistant": "gpt"
}

# === WRAPPER  CLASS ===================================================================================================
class Wrapper:
    """
    Wraps around model
    """

    def __init__(
        self,
        model_path: Path,
        lora_path: Path = None,
        max_seq_length: int = 512
    ):
        """
        Default constructor
        :param model_path: Path to checkpoint
        :param lora_path: Path to LoRA
        :param max_seq_length: max seq length
        """

        # Initialize model
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True
        )

        # If lora path provided, load it
        if lora_path is not None:
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                lora_path)

    def inference(
            self,
            message_history: list[dict[str:str]],
            temperature: float = 1.0,
            max_new_tokens=MAX_NEW_TOKENS
    ):
        """
        Do inference
        :param message_history: Message history
        :param temperature: Inference temperature
        :param max_new_tokens: Max number of new tokens in response
        :return: String response
        """

        # Tokenize inputs
        input_ids = self.tokenizer.apply_chat_template(
            message_history,
            return_tensors="pt"
        ).to(self.model.device)

        # Set terminators
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate outputs
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )

        # Parse output
        response = outputs[0][input_ids.shape[-1]:]
        response_message = self.tokenizer.decode(response, skip_special_tokens=True)

        # Return response
        return response_message

    # --- Train a LoRA -------------------------------------------------------------------------------------------------
    def train_lora(
            self,
            output_path: Path,  # File to output to
            dataset_path: Path,  # Location of dataset
            max_seq_length: int = MAX_SEQ_LENGTH,  # Max sequence length
            dataset_mapping: dict[str: str] = DATASET_MAPPING  # Input file work mappings
    ):
        # Replace model with LoRA adapter
        self.model = unsloth.FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=16,
            lora_dropout=0,  # Don't change - 0 is optimized
            bias="none",  # Don't change = 'none' is optimized
            use_gradient_checkpointing="unsloth",  # Don't change obviously
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )

        # Replace tokenizer with mapping tokenizer
        self.tokenizer = unsloth.get_chat_template(
            self.tokenizer,
            chat_template="llama-3",
            mapping=dataset_mapping
        )

        # Function to format prompt
        def format_as_prompt(examples):

            # Get  conversations
            conversations = examples["conversations"]

            # Apply chat template and mappings
            text = [self.tokenizer.apply_chat_template(conversation,
                                                       tokenize=False,
                                                       add_generation_prompt=False)
                    for conversation in conversations]

            # Return result
            return {"text": text}

        # Open dataset
        dataset = datasets.load_dataset(
            str(dataset_path),
            split="train"
        )

        # Map dataset
        dataset = dataset.map(
            format_as_prompt,
            batched=True
        )

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,  # TODO check - Can make training 5x faster for short sequences.
            args=transformers.TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not unsloth.is_bfloat16_supported(),
                bf16=unsloth.is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            ),
        )

        # Do training
        trainer_stats = trainer.train()
        print(trainer_stats)

        # Save model
        self.model.save_pretrained_merged(output_path, self.tokenizer, save_method="lora")