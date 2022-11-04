from bert.lib import *
from config import *

def predict(device, tokenizer, model, review_text):
    print(review_text)
    # review_text = "東北大学で地震の研究をしています。"

    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=max_length,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )

    print(encoded_review)
    print("=====================")
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    print(output)
    _, prediction = torch.max(output, dim=1)

    labelVal = class_names[prediction]
    print(torch.max(output, dim=1))
    print(f'Review text: {review_text}')
    print(f'class_names : {class_names}')
    print(f'prediction : {prediction}')
    print(f'_ : {_}')
    print(f'Label Value  : {labelVal}')

    return labelVal
# if __name__ == "__main__":
#     predict()