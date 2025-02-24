import json
import torch
import regex as re
import xgrammar as xgr
import PIL
import os

from PIL import Image
from tqdm import tqdm
from PIL import Image
from vllm import SamplingParams
from torch.utils.data import DataLoader
from typing import Dict, List
from collections import defaultdict
from transformers import AutoConfig
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
    
    def get_xgrammar_compiler(self, tokenizer, model_name: str) -> xgr.GrammarCompiler:
        """
        Return's Grammar Compiler based on tokenizer info -> needed for Logits Processor
        during generation process
        """
        config = AutoConfig.from_pretrained(model_name)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
        grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

        compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
        return compiled_grammar
    
    def get_all_images(self, path: str) -> List:
        all_imgs: List = []
        for elem in path:
            img_path: str = os.path.join(path, elem)
            try:
                img = Image.open(img_path).convert("RGB")
                all_imgs.append(img)
            except Exception as e:
                print(f"[get_all_images] Błąd wczytywania obrazu {img_path}: {e}")

        return all_imgs

    def generate_json_from_model(
            self, 
            example, 
            model, 
            debug: bool = True,
            use_xgrammar: bool = False,
        ):
        curr_path: str = example["subfolder_name"]
        list_of_images: List = self.get_all_images(curr_path)
        prompt: str = """ 
        Make a one, hierarchical .json from the images. Combine it with other messages. 
        Leave only generated structure, which will be dumped in the future
        """

        sampling_params = SamplingParams(temperature=0.00)
        with torch.no_grad():
            outputs = model.generate({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": list_of_images
                },
            }, sampling_params)

        generated_text: str = ""
        for o in outputs:
            generated_text = o.outputs[0].text

        if debug:
            print(f"Generated text: {generated_text}")

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
    
    def normalize_json_str(self, json_str: str) -> str:
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
            vLLM_model,
            model_fix,
            processor_fix,
            do_auto_fix: bool = False,
            use_xgrammar: bool = False,
            do_normalize_jsons: bool = True,
            debug: bool = False
    ) -> tuple:
        """
        Zwraca krotkę (artefact_percentage, valid_after_clean_percentage).
        
        Jeśli do_auto_fix=True, dla niepoprawnych JSONów zostanie podjęta próba naprawy przy użyciu model_fix i tokenizer_fix.
        """
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        if use_xgrammar:
            print(f"\nUsing XGrammar as backend\n")

        pages_lev_map = defaultdict(lambda: [0, 0])

        count_artefacts = 0
        count_valid_after_clean = 0
        num_examples = len(test_set)
        lev_sum = 0
        lev_count = 0

        if num_examples == 0:
            print("[evaluate_on_testset] Brak przykładów w test_secie.")
            return 0.0, 0.0
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            if not isinstance(batch, list):
                batch = [batch]

            for example in batch:
                message_list = example["messages"]
                page_count = sum(
                    1 for m in message_list 
                    if isinstance(m, dict) and m.get("type") == "image"
                )

                pred_json_str = self.generate_json_from_model(
                    example, 
                    vLLM_model,
                    debug=debug, 
                    use_xgrammar=use_xgrammar
                )
                
                ground_json: str = self.load_ground_json(example = example)

                if self.check_if_any_artefacts(pred_json_str):
                    count_artefacts += 1

                cleaned_str = self.extract_clean_json(pred_json_str)

                if do_auto_fix and not self.is_json_loadable(cleaned_str):
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

                            pages_lev_map[page_count][0] += dist_val
                            pages_lev_map[page_count][1] += 1
                        else:
                            dist_val = distance(cleaned_str, ground_json)
                            lev_sum += dist_val
                            lev_count += 1

                            pages_lev_map[page_count][0] += dist_val
                            pages_lev_map[page_count][1] += 1

                del batch
                torch.cuda.empty_cache()

        avg_lev_dist: float = 0.0
        if lev_count > 0:
            avg_lev_dist = lev_sum / lev_count

        artefact_percentage = round((count_artefacts / num_examples) * 100, 2)
        valid_after_clean_percentage = round((count_valid_after_clean / num_examples) * 100, 2)
        return artefact_percentage, valid_after_clean_percentage, avg_lev_dist, pages_lev_map