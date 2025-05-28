from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# --- 0. Configuration ---
MODEL_NAME = "facebook/opt-350m"
RESPONSE_TEMPLATE_STR = "[START_RESPONSE]"

# --- 1. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# --- 2. Prepare your Dataset ---
raw_data = [
    {
        "prompt": "[COURSE] English [CHAT] User: Hi there Assistant: Hello! [QUERY] What is he...",
        "resolved_query_tag": "[RESOLVED_QUERY']",
        "completion": "Karan Taneja is an AI researcher focusing on LLMs."
    },
    {
        "prompt": "[COURSE] History [CHAT] User: Tell me about Napoleon Assistant: Sure! [QUERY] When was he born...",
        "resolved_query_tag": "[RESOLVED_QUERY']",
        "completion": "Napoleon Bonaparte was born on August 15, 1769."
    },
    {
        "prompt": "[COURSE] Science [CHAT] User: What is photosynthesis? Assistant: It's a process... [QUERY] What are the reactants...",
        "resolved_query_tag": "[RESOLVED_QUERY']",
        "completion": "The reactants in photosynthesis are carbon dioxide, water, and sunlight."
    }
]

def format_example(example):
    full_prompt = f"{example['prompt']}{example['resolved_query_tag']}"
    return f"{full_prompt}{RESPONSE_TEMPLATE_STR}{example['completion']}"

processed_texts = [format_example(example) for example in raw_data]
train_dataset_dict = {"text": processed_texts} # Your dataset has a "text" column
train_dataset = Dataset.from_dict(train_dataset_dict)

print("\nSample formatted training text:")
print(train_dataset[0]['text'])

# --- 3. Initialize Data Collator ---
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE_STR,
    tokenizer=tokenizer
)
print(f"\nUsing DataCollatorForCompletionOnlyLM with response_template: '{RESPONSE_TEMPLATE_STR}'")

# --- 4. SFTTrainer Configuration ---
training_args = SFTConfig(
    output_dir="/tmp/sft_resolved_query_example",
    max_seq_length=128,
    per_device_train_batch_size=1,
    max_steps=20,
    logging_steps=1,
    report_to=[],
    dataset_num_proc=None, # Explicitly setting for some SFTConfig versions
)

# --- 5. Initialize SFTTrainer ---
# MODIFICATION: Removed 'dataset_text_field'
# Assuming the trainer will default to the 'text' column in your train_dataset
trainer = SFTTrainer(
    MODEL_NAME,  # Model identifier as the first positional argument
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
    # dataset_text_field="text" # REMOVED THIS LINE
    # tokenizer=tokenizer, # Kept removed
)
print("\nSFTTrainer initialized.")

# --- 6. Train ---
print("\nStarting training...")
trainer.train()
print("Training finished.")

# --- 7. Conceptual Verification (simplified) ---
print("\n--- Conceptual Verification ---")
sample_idx = 0
text_sample = train_dataset[sample_idx]['text']
tokenized_output = tokenizer(text_sample)
input_ids = tokenized_output.input_ids

print(f"\nOriginal Text Sample:\n{text_sample}")
print(f"\nTokenized Input IDs ({len(input_ids)} tokens):\n{input_ids}")
print(f"Decoded Input IDs:\n{tokenizer.decode(input_ids)}")

response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE_STR, add_special_tokens=False)
print(f"\nTokenized Response Template '{RESPONSE_TEMPLATE_STR}':\n{response_template_ids}")

labels = list(input_ids)
found_template = False
for i in range(len(labels) - len(response_template_ids) + 1):
    if labels[i:i+len(response_template_ids)] == response_template_ids:
        for j in range(i):
            labels[j] = -100
        found_template = True
        break

if not found_template:
    print("WARNING: Response template not found in the sample.")

print(f"\nConceptual Labels for this sample (before padding/truncation in a batch):\n{labels}")
print("\nTokens that will contribute to loss (decoded from labels where label != -100):")
predicted_tokens = [tok_id for tok_id in labels if tok_id != -100]
if predicted_tokens:
    print(tokenizer.decode(predicted_tokens))
else:
    print("No tokens marked for prediction.")