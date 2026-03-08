import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class TextInference:

    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "models/text_model"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "models/text_model"
        )

        self.model.eval()

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)

            confidence, predicted = torch.max(probs, dim=1)

        sentiment = predicted.item()

        return sentiment, confidence.item()