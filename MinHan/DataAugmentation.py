import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the model and tokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Function to get paraphrased response
def get_response(input_text, num_return_sequences, num_beams):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

# Function to paraphrase a DataFrame column
def paraphrase_column(df, column_name, num_return_sequences=1, num_beams=10):
    paraphrased_column = []
    for text in df[column_name]:
        if pd.notna(text) and text.strip():  # Check if the text is not NaN and not empty
            # print("Original: " + text)
            paraphrased_text = get_response(text, num_return_sequences, num_beams)
            # print("Paraphrased: ", paraphrased_text)
            paraphrased_column.append(paraphrased_text[0])  # Take the first paraphrased sequence
        else:
            paraphrased_column.append('')  # Append empty string for NaN or empty text
    return paraphrased_column

# Load the CSV file
csv_file_path = '../Main/Cleansed_Data/cleansed_data.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Select only the first 20 rows (EDIT TO ROWS NEEDED)
df = df.head(20)

# Select the specific columns
columns_to_paraphrase = ['deficiency_finding', 'description_overview', 'immediate_causes', 'root_cause_analysis', 'corrective_action', 'preventive_action']

# Paraphrase the content in the selected columns
for column in columns_to_paraphrase:
    df[column] = paraphrase_column(df, column)

# Save the paraphrased DataFrame to a new CSV file
output_csv_path = 'augmented_data.csv'
df.to_csv(output_csv_path, index=False)

print(f"Paraphrased data for the first 20 rows saved to {output_csv_path}")