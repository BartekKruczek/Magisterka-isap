import torch

from json_handler import JsonHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen2Half(JsonHandler):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.xlsx_path = "matching_dates_cleaned.xlsx"
        self.model_variant = "Qwen/Qwen2.5-72B-Instruct"

        if self.device.type == "cuda":
            self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        elif self.device.type == "mps" or self.device.type == "cpu":
            self.cache_dir = "/Users/bk/Documents/Zajęcia (luty - czerwiec 2024)/Pracownia-problemowa/.cache"

        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Qwen2.5"
    
    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
        self.model_variant,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        cache_dir = self.cache_dir,
        attn_implementation = "flash_attention_2",
        )

        return model
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_variant)

        return tokenizer
    
    def get_dataset(self, combined_string: str = None) -> list[dict]:
        prompt = f"Can you combine three separate json files into one? All files had been created from one law document. Text to combine: {combined_string}. Leave only generated structure, polish language."
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        return messages
    
    def get_response(self, messages: list[dict] = None) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens = 32768,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def save_combined_json(self) -> None:
        try:
            json_text = self.get_response(self.get_dataset(self.json_load(path = "JSON_files")))
            self.json_dump(json_text, idx = 0)
        except Exception as e:
            print(f"Error occurred in {self.save_combined_json.__name__}, error: {e}")