from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Completer:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def complete(self, input_text):
        input_ids = self.tokenizer.encode("complete: " + input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
