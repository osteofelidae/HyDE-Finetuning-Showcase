# === INPUT FIELD ======================================================================================================
PROMPT = """
### RemoteEvent returning nil 
Hello. I am currently scripting a speedrun leaderboard but I have run into a problem with remote events. I have a server script for the leaderboard and a local script with the timer information. Currently, the server script is firing a remote event to which the local script receives and supposedly should return a string, however, it returns nil.

As you can see below the error, the time is being printed but not returned. Here are the functions.

Server script:
```
new.Value.Text = game.ReplicatedStorage.ReadableTimer:FireAllClients(d[2]/1000)
```

Local script:
```
game.ReplicatedStorage.ReadableTimer.OnClientEvent:Connect(function(timertime)
	print(readabletime(timertime))
	return readabletime(timertime)
end)
```

What did I do wrong?
"""


# === DESCRIPTION ======================================================================================================
"""
LLM classes & methods
"""
import re
from json import JSONDecoder

# === DEPENDENCIES =====================================================================================================
import torch
import transformers
import peft
import unsloth
import datasets
from src.utils.path_utils import path
from pathlib import Path
from trl import SFTTrainer
from data.roblox_docs_dumps.prepare import filter_links



# === CONSTANTS ========================================================================================================
MODEL_PATH = path("models/llama-3-8b-instruct")
MAX_NEW_TOKENS = 1024
MAX_SEQ_LENGTH = 1024
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
        model_path: Path = MODEL_PATH,
        lora_path: Path = None,
        max_seq_length: int = MAX_SEQ_LENGTH
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
        return response_message.replace("assistant\n", "").strip()

    def train_lora(
            self,
            output_path: Path,  # File to output to
            dataset_path: str | list[str],  # Location of dataset
            max_seq_length: int = MAX_SEQ_LENGTH,  # Max sequence length
            dataset_mapping: dict[str: str] = DATASET_MAPPING,  # Input file work mappings
            lora_only: bool = True
    ):
        """
        Train and save a LoRA adapter
        :param output_path: LoRA output path
        :param dataset_path: Training set path
        :param max_seq_length: Duh
        :param dataset_mapping: Mapping for dataset (eg. ShareGPT format)
        :param lora_only: Whether to save lora only
        :return:
        """

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
            "json",
            data_files=dataset_path,
            split="train"
        )
        dataset.shuffle()

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
                per_device_train_batch_size=1,
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
                num_train_epochs=3
            ),
        )

        # Do training
        trainer_stats = trainer.train()
        print(trainer_stats)

        # Save model
        if lora_only:
            self.model.save_pretrained_merged(output_path, self.tokenizer, save_method="lora")
        else:
            self.model.save_pretrained_merged(output_path, self.tokenizer)

    def generate_embedding(
            self,
            text
    ):
        """
        Generate embedding for some text
        :param text: Text.
        :return: Embeddings.
        """

        # Tokenize inputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.logits.mean(dim=1)
        return embedding


def save_embedding(
        embedding,
        output_path: Path
):
    """
    Save embedding to file
    :param embedding: The embedding.
    :param output_path: File path
    :return: None
    """

    torch.save(embedding, output_path)

def load_embedding(
        input_path: Path
):
    """
    Load embedding from file
    :param input_path: File path
    :return: None
    """

    return torch.load(input_path)

def compare_embeddings(
    embedding_1,
    embedding_2
):
    return torch.nn.functional.cosine_similarity(embedding_1, embedding_2, dim=1).item()

def generate_dummy_response(
        wrapper: Wrapper,
        prompt: str,
        info=""
):
    message_history = []

    message_history.append({
        "role": "system",
        "content": "You are a teacher. Provide vague hints to help answer the question. Use the information provided if necessary. Do not provide code."
    })

    message_history.append({
        "role": "system",
        "content": info
    })

    message_history.append({
        "role": "user",
        "content": prompt
    })
    response = wrapper.inference(
        message_history=message_history,
        temperature=0.1
    )
    return response

def generate_keywords(
        wrapper: Wrapper,
        prompt: str
):
    message_history = []
    message_history.append({
        "role": "system",
        "content": "You are a helpful AI assistant. Provide a short, comma-separated list of Roblox documentation page titles that may be helpful to the request. Provide your answer in the form 'item 1, item 2, ...' up to 5 items. Provide only the list with no introduction or label. Do not respond to the request directly."
    })
    message_history.append({
        "role": "user",
        "content": prompt
    })
    response = wrapper.inference(
        message_history=message_history,
        temperature=0.1
    )
    return [res.replace("assistant", "").replace("Roblox documentation page titles: ", "").strip() for res in response.split(",") if not "roblox" in res]

def generate_dummy_article(
        wrapper: Wrapper,
        prompt: str
):
    message_history = []
    message_history.append({
        "role": "system",
        "content": "You are a helpful AI assistant. Respond to the question about Roblox. Provide code examples where necessary. Only write one paragraph."
    })
    message_history.append({
        "role": "user",
        "content": f"What is '{prompt}' in Roblox?"
    })
    response = wrapper.inference(
        message_history=message_history,
        temperature=1.0
    )
    return response






# === MAIN =============================================================================================================
if __name__ == "__main__":

    # Instantiate model
    wrapper = Wrapper(
        model_path=path("models/llama-3-8b-instruct_pretrained_4"),
    )
    inp = PROMPT

    words = generate_keywords(wrapper, inp)[:5] + [inp]
    #words = [inp]
    print(words)

    dummies = [generate_dummy_article(wrapper, word) for word in words]

    similar_dict = {}
    source = path("data/roblox_docs_embeddings")
    pool = [x for x in source.iterdir() if not "-" in x.name]

    for dummy in dummies:
        target = wrapper.generate_embedding(text=dummy)
        for file in pool:
            if file.is_file() and file.name.endswith(".pt"):
                try:
                    comp = compare_embeddings(embedding_1=load_embedding(file), embedding_2=target)
                    if file.stem not in similar_dict or similar_dict[file.stem] < comp:
                        similar_dict[file.stem] = comp
                except:
                    print("err")

    similar = []
    for i in range(5):
        item = max(similar_dict, key=similar_dict.get)
        similar.append(item)
        del similar_dict[item]

    print("FINAL: " + str(similar))

    text = "# INFO:\n"
    for name in similar:
        with open(path("data/roblox_docs_cleaned")/f"{name}.md") as file:
            text += f"## {name}\n{file.read().split('#')[0]}\n\n"
    text += "# END OF INFO"

    res = generate_dummy_response(
        wrapper=wrapper,
        info=text,
        prompt=inp)

    print(res)
    print("=========")
    res = wrapper.inference(
        message_history=[
            {
                "role": "system",
                "content": "Rewrite the prompt so that it makes sense without code. Do NOT include any example code. Use proper grammar."
            },
            {
                "role": "user",
                "content": res
            }
        ],
    )

    with open("log.txt", "a") as file:
        file.write(f"\n\n{'='*100}\n{inp}\n{'-'*100}\n{res}")
    print(res)