"""
IndoBERT Model for Javanese Emotion Classification
"""
import torch
from transformers import AutoModel, AutoTokenizer


class IndoBERTClassifier(torch.nn.Module):
    """     
    IndoBERT-based classifier for emotion detection
    Architecture: BERT -> Dropout -> FC1 -> ReLU -> Dropout -> FC2
    """
    def __init__(self, num_classes=4, model_name='indolem/indobert-base-uncased'):
        super(IndoBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(768, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.dropout(pooled)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def bert_encode(texts, tokenizer, max_len=128):
    """
    Encode texts using BERT tokenizer
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer
        max_len: Maximum sequence length
    
    Returns:
        input_ids: Tensor of token IDs
        attention_masks: Tensor of attention masks
    """
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks