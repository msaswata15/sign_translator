from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class T5Completer:
    def __init__(self, model_name='t5-small', device='cpu', max_length=64):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device
        self.max_length = max_length

    def complete(self, partial_sentence: str) -> str:
        input_text = f"complete: {partial_sentence}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=4,
            early_stopping=True
        )
        completed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completed_text

    def correct(self, text, max_length=128):
        prompt = f"fix: {text}"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected