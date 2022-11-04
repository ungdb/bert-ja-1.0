from bert.lib import *
from config import *
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese', truncation=True)
# model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', return_dict=True).to(device)

# nlp = spacy.load("ja_core_news_sm")

def sentence_classification(device, tokenizer, model, nlp, text):
    # text = "て愛愛してます。"
    # sents = text
    sents = [sent.text for sent in nlp(text).sents]
    sents = sents*2

    encoded_data_val = tokenizer.encode_plus(
        sents, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding='longest',
        truncation=True,
        max_length=max_length, 
        return_tensors='pt'
    )

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']

    dataset_val = TensorDataset(input_ids_val, attention_masks_val)

    batch_size = 4
    dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

    all_logits = np.empty([0,2])

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                }

        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[0]
        all_logits = np.vstack([all_logits, torch.softmax(logits, dim=1).detach().cpu().numpy()])

        # print(logits)
        # print(all_logits)
        # print(np.max(all_logits))
        logits_max = np.max(all_logits)

        # with torch.no_grad():
        # logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()

        labels = model.config.id2label[predicted_class_id]
        return predicted_class_id, labels, logits_max
        # print(labels)