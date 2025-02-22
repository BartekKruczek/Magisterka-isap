import json
import torch
import regex as re

from tqdm import tqdm
from typing import Dict
from qwen_vl_utils import process_vision_info
from Levenshtein import distance

from json_handler import JsonHandler

class CustomMetrics(JsonHandler):
    def __init__(self) -> None:
        super().__init__()

    def load_ground_json(self, example: Dict) -> str:
        if not example:
            print(f"Ground truth JSON path not found")
        
        path: str = example["json_ground_path"]
        json_str: str = None

        with open(path, "r", encoding="utf-8") as f:
            json_str = f.read()

        return json_str

    def generate_json_from_model(
            self, 
            example, 
            model, 
            processor, 
            max_new_tokens=8192, 
            debug: bool = False,
        ):
        message = example["messages"]
        
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        if debug:
            print("[generate_json_from_model] Prompt:\n", prompt_str)

        image_inputs, _ = process_vision_info(message)

        if image_inputs is not None:
            inputs = processor(
                text=[prompt_str],
                images=[image_inputs],
                return_tensors="pt",
                padding=True
            )
        else:
            inputs = processor(
                text=[prompt_str],
                return_tensors="pt",
                padding=True
            )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
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

        has_before = bool(match_before and match_before.group(1).strip())
        has_after = bool(match_after and match_after.group(1).strip())

        return has_before or has_after
    
    def normalize_json_str(json_str: str) -> str:
        """
        Parse a JSON string into a Python object, sort all dictionaries,
        then dump back to a string with consistent settings.
        Returns the normalized JSON string if successful, otherwise the original string.
        """
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return json_str

        def sort_dict_keys(item):
            if isinstance(item, dict):
                return {k: sort_dict_keys(item[k]) for k in sorted(item)}
            elif isinstance(item, list):
                return [sort_dict_keys(elem) for elem in item]
            else:
                return item

        sorted_obj = sort_dict_keys(obj)

        normalized_str = json.dumps(sorted_obj, ensure_ascii=False, separators=(",", ":"))
        return normalized_str
    
    def extract_clean_json(self, s: str) -> str:
        """
        Zwraca wycinek ciągu s od pierwszego '{' do ostatniego '}' włącznie.
        Jeśli się nie powiedzie, zwraca cały ciąg.
        """
        start_idx = s.find('{')
        end_idx = s.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return s
        return s[start_idx:end_idx+1]

    def is_json_loadable(self, s: str) -> bool:
        """
        Sprawdza, czy dany ciąg s da się wczytać jako poprawny JSON.
        """
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False

    def auto_fix_json(self, s: str, model, processor, max_iterations: int = 5, debug: bool = False) -> str:
        """
        Iteracyjnie próbuje poprawić niepoprawny ciąg JSON przy użyciu modelu fixera.
        Maksymalnie wykonuje max_iterations prób. Do promptu dodaje komunikat błędu z poprzedniej próby.
        """
        candidate = s
        for i in range(max_iterations):
            try:
                json.loads(candidate)
                if debug:
                    print(f"[auto_fix_json] Iteracja {i}: JSON jest poprawny.")
                return candidate
            except json.JSONDecodeError as e:
                error_str = str(e)
                if debug:
                    print(f"[auto_fix_json] Iteracja {i}: JSON niepoprawny, błąd: {error_str}. Próba naprawy.")
                fix_prompt = (
                    "Popraw poniższy niepoprawny JSON tak, aby był poprawnym JSON-em. "
                    "Zwróć wyłącznie poprawny JSON bez dodatkowych komentarzy.\n\n"
                    f"Błąd: {error_str}\n\n"
                    "Niepoprawny JSON:\n" + candidate
                )
                # Tworzymy przykładową strukturę wiadomości
                example_fix = {"messages": [{"role": "user", "content": fix_prompt}]}
                candidate = self.generate_json_from_model(example_fix, model, processor, debug=debug)
            if debug:
                print("[auto_fix_json] Maksymalna liczba iteracji osiągnięta. Zwracam ostatni wynik.")
        return candidate

    def evaluate_on_testset(
            self, 
            test_set,
            model, 
            processor,
            model_fix,
            processor_fix,
            do_auto_fix: bool = False,
            do_normalize_jsons: bool = True,
            debug: bool = False
    ) -> tuple:
        """
        Zwraca krotkę (artefact_percentage, valid_after_clean_percentage).
        
        Jeśli do_auto_fix=True, dla niepoprawnych JSONów zostanie podjęta próba naprawy przy użyciu model_fix i tokenizer_fix.
        """
        count_artefacts = 0
        count_valid_after_clean = 0
        num_examples = len(test_set)
        lev_sum = 0
        lev_count = 0

        if num_examples == 0:
            print("[evaluate_on_testset] Brak przykładów w test_secie.")
            return 0.0, 0.0

        for example in tqdm(test_set, desc="Evaluating", total=num_examples):
            pred_json_str = self.generate_json_from_model(example, model, processor, debug=debug)
            ground_json: str = self.load_ground_json(example = example)

            if self.check_if_any_artefacts(pred_json_str):
                count_artefacts += 1

            cleaned_str = self.extract_clean_json(pred_json_str)

            if do_auto_fix:
                if not self.is_json_loadable(cleaned_str):
                    cleaned_str = self.auto_fix_json(cleaned_str, model_fix, processor_fix, max_iterations=5, debug=debug)

            if self.is_json_loadable(cleaned_str):
                count_valid_after_clean += 1

                if ground_json:
                    if do_normalize_jsons:
                        predicted_norm = self.normalize_json_str(cleaned_str)
                        ground_norm = self.normalize_json_str(ground_json)

                        dist_val = distance(predicted_norm, ground_norm)
                        lev_sum += dist_val
                        lev_count += 1
                    else:
                        dist_val = distance(cleaned_str, ground_json)
                        lev_sum += dist_val
                        lev_count += 1

        avg_lev_dist: float = 0.0
        if lev_count > 0:
            avg_lev_dist = lev_sum / lev_count

        artefact_percentage = round((count_artefacts / num_examples) * 100, 2)
        valid_after_clean_percentage = round((count_valid_after_clean / num_examples) * 100, 2)
        return artefact_percentage, valid_after_clean_percentage, avg_lev_dist