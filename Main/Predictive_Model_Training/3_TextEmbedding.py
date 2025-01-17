from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def get_bert_embeddings(df):
    torch.cuda.empty_cache()
    """
    Generate BERT embeddings for text columns.
    """
    # Combine text columns into a single column
    text_columns = ['deficiency_finding', 'description_overview', 'immediate_causes', 'root_cause_analysis', 'VesselGroup']
    df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')

    # Function to get BERT embeddings in smaller batches
    def _get_embeddings(texts, batch_size=16, max_length=512):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

    # Get BERT embeddings for the combined text
    text_embeddings = _get_embeddings(df['combined_text'].tolist(), batch_size=16, max_length=512)

    return text_embeddings