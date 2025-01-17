import torch  # type: ignore
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from torch.cuda.amp import autocast  # For mixed precision (FP16)

# Load the model and tokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Function to get paraphrased response
def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    with autocast():  # Enable mixed precision (FP16) if GPU supports it
        translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

# Function to paraphrase a DataFrame column in batches
def paraphrase_column(df, column_name, num_return_sequences=1, num_beams=10, batch_size=16):
    paraphrased_column = []
    texts = df[column_name].tolist()
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        batch = [text if pd.notna(text) and text.strip() else '' for text in batch]  # Replace empty/NaN with empty string
        if batch:
            # Tokenize and generate paraphrases for the batch
            tokenized_batch = tokenizer(batch, truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
            with autocast():  # Enable mixed precision (FP16) if GPU supports it
                translated = model.generate(**tokenized_batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
            paraphrased_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
            paraphrased_column.extend(paraphrased_batch)
        else:
            paraphrased_column.extend([''] * len(batch))  # Append empty strings for empty or NaN texts
        i += batch_size
        print(f"Processed {i} out of {len(texts)} rows in column '{column_name}'")
    return paraphrased_column

# Load the CSV file
csv_file_path = './Main/Cleansed_Data/predicted_severity.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Select 2500 random rows
random_sample = df.sample(n=4000, random_state=42)  # random_state ensures reproducibility

# Reset the index of the sampled DataFrame
df = random_sample.reset_index(drop=True)

# Select the specific columns to paraphrase
columns_to_paraphrase = [
    'deficiency_finding', 
    'description_overview', 
    'immediate_causes', 
    'root_cause_analysis', 
    'corrective_action', 
    'preventive_action'
]

# Paraphrase the content in the selected columns
for column in columns_to_paraphrase:
    print(f"Paraphrasing column: {column}")
    df[column] = paraphrase_column(df, column)

# Save the paraphrased DataFrame to a new CSV file
output_csv_path = './Main/Cleansed_Data/augmented_data2.csv'
df.to_csv(output_csv_path, index=False)

print(f"Paraphrased data saved to {output_csv_path}")