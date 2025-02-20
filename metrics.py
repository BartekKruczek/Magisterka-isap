import json
import torch
import regex as re

from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from Levenshtein import distance

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
            do_auto_fix: bool = False,
            debug: bool = False
    ) -> tuple:
        """
        Zwraca krotkę (artefact_percentage, valid_after_clean_percentage).
        
        Jeśli do_auto_fix=True, dla niepoprawnych JSONów zostanie podjęta próba naprawy przy użyciu model_fix i tokenizer_fix.
        """
        count_artefacts = 0
        count_valid_after_clean = 0
        num_examples = len(test_set)

        if num_examples == 0:
            print("[evaluate_on_testset] Brak przykładów w test_secie.")
            return 0.0, 0.0

        for example in tqdm(test_set, desc="Evaluating", total=num_examples):
            pred_json_str = self.generate_json_from_model(example, model, processor, debug=debug)

            # 1. Sprawdzamy artefakty
            if self.check_if_any_artefacts(pred_json_str):
                count_artefacts += 1

            # 2. Wycinamy fragment z JSON-em
            cleaned_str = self.extract_clean_json(pred_json_str)

            # 3. Jeśli auto-fix jest włączony, a JSON nie parsuje, spróbuj poprawić
            if do_auto_fix:
                if not self.is_json_loadable(cleaned_str):
                    cleaned_str = self.auto_fix_json(cleaned_str, model, processor, max_iterations=5, debug=debug)

            # 4. Sprawdzamy, czy wynikowy ciąg da się sparsować jako JSON
            if self.is_json_loadable(cleaned_str):
                count_valid_after_clean += 1

        artefact_percentage = round((count_artefacts / num_examples) * 100, 2)
        valid_after_clean_percentage = round((count_valid_after_clean / num_examples) * 100, 2)
        return artefact_percentage, valid_after_clean_percentage