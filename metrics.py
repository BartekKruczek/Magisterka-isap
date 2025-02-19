import json
import torch
import regex as re
from tqdm import tqdm  # <-- dodanie importu tqdm

from qwen_vl_utils import process_vision_info
from json_handler import JsonHandler

class CustomMetrics(JsonHandler):
    def __init__(self) -> None:
        super().__init__()

    def generate_json_from_model(
            self, 
            example, 
            model, 
            processor, 
            max_new_tokens=8192, 
            debug: bool = False,
        ):
        """
        Generuje tekstową odpowiedź modelu (zwykle JSON), bazując na wejściu (example).
        Zwraca surowy ciąg tekstowy wygenerowany przez model.
        """
        message = example["messages"]
        
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        if debug:
            print("[generate_json_from_model] Prompt:\n", prompt_str)

        image_inputs, _ = process_vision_info(message)

        inputs = processor(
            text=[prompt_str],
            images=[image_inputs],
            return_tensors="pt",
            padding=True
        )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_beams=1,
            temperature=0.01,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if debug:
            print("[generate_json_from_model] Wygenerowany tekst:\n", generated_text)

        return generated_text
    
    def check_if_any_artefacts(self, s: str) -> bool:
        """
        Sprawdza, czy w przekazanym ciągu s występują jakiekolwiek znaki 
        przed pierwszym '{' lub po ostatnim '}'.
        Zwraca True, jeśli artefakty istnieją, w przeciwnym razie False.
        """
        match_before = re.search(r'^(.*?){', s, flags=re.DOTALL)
        match_after = re.search(r'}(.*?)$', s, flags=re.DOTALL)

        # Jeśli cokolwiek występuje w group(1) przed '{' lub po '}', mamy artefakty
        has_before = bool(match_before and match_before.group(1).strip())
        has_after = bool(match_after and match_after.group(1).strip())

        return has_before or has_after
    
    def extract_clean_json(self, s: str) -> str:
        """
        Zwraca wycinek ciągu s od pierwszego '{' do ostatniego '}' włącznie.
        Jeśli się nie powiedzie, zwraca cały ciąg.
        """
        start_idx = s.find('{')
        end_idx = s.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return s  # nie udało się poprawnie wyekstrahować
        return s[start_idx:end_idx+1]

    def is_json_loadable(self, s: str) -> bool:
        """
        Sprawdza, czy dany ciąg (s) da się wczytać jako poprawny JSON.
        """
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    def evaluate_on_testset(
            self, 
            test_set,
            model, 
            processor,
    ) -> tuple:
        """
        Zwraca krotkę (artefact_percentage, valid_after_clean_percentage).

        artefact_percentage       -> % przykładów, które posiadały artefakty 
                                     (znaki przed pierwszym '{' lub po ostatnim '}')
        valid_after_clean_percentage -> % przykładów, które po wycięciu fragmentu 
                                        od '{' do '}' dają się zdeserializować (json.loads).
        """
        count_artefacts = 0
        count_valid_after_clean = 0
        num_examples = len(test_set)

        if num_examples == 0:
            print("[evaluate_on_testset] Brak przykładów w test_secie.")
            return 0.0, 0.0

        # Dodanie paska postępu
        for example in tqdm(test_set, desc="Evaluating", total=num_examples):
            pred_json_str = self.generate_json_from_model(example, model, processor)

            # 1. Sprawdzamy artefakty
            if self.check_if_any_artefacts(pred_json_str):
                count_artefacts += 1

            # 2. Wyczyszczony ciąg
            cleaned_str = self.extract_clean_json(pred_json_str)

            # 3. Sprawdzamy, czy można go wczytać jako poprawny JSON
            if self.is_json_loadable(cleaned_str):
                count_valid_after_clean += 1

        artefact_percentage = round((count_artefacts / num_examples) * 100, 2)
        valid_after_clean_percentage = round((count_valid_after_clean / num_examples) * 100, 2)

        return artefact_percentage, valid_after_clean_percentage