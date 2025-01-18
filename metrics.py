import zss
import torch

from qwen_vl_utils import process_vision_info
from json_handler import JsonHandler
from node_creator import CustomNodeCreator

class CustomMetrics(JsonHandler):
    def __init__(self) -> None:
        super().__init__()

    def calculate_tree_edit_distance(self, json_generated: str = None, json_test_path: str = None, debug: bool = True) -> int:
        json_generated: str = json_generated
        json_test: str = self.json_load_TED(json_file2_path = json_test_path)

        if json_generated is None or json_test is None:
            print(f'Error loading json files in {self.calculate_tree_edit_distance.__name__}')
            return None

        # debug
        if debug:
            print(f'Json 1 as dictionary: {str(json_generated)} \n')
            print(f'Json 2 as dictionary: {str(json_test)} \n')

        # creating tree section
        tree1 = self.create_tree_from_json_string(json = json_generated)
        tree2 = self.create_tree_from_json_string(json = json_test)

        # distance section
        calc_dist: int = zss.simple_distance(
            A = tree1,
            B = tree2,
            get_children = lambda my_node: my_node.get_children(),
            get_label = lambda my_node: my_node.get_label(),
            label_dist = lambda label1, label2: 0 if label1 == label2 else 1
        )

        return calc_dist
    
    def count_nodes_in_tree(self, node: CustomNodeCreator) -> int:
        if not node:
            return 0
        total = 1
        for child in node.get_children():
            total += self.count_nodes_in_tree(child)
        return total
    
    def ted_based_accuracy(self, json_pred_str: str, json_gt_str: str, debug: bool = False) -> float:
        """
        TED accuracy = max(0, 1 - (TED / maxTED)), maxTED = liczba węzłów w drzewie ground-truth (tj. porównanie z drzewem pustym).
        """
        ted_distance = self.calculate_tree_edit_distance(json_pred_str, json_gt_str, debug = debug)
        if ted_distance is None:
            print(f"Error calculating TED in {self.ted_based_accuracy.__name__}, returning 0.0")
            return 0.0

        # maxTED section
        tree_gt = self.create_tree_from_json_string(json_gt_str)
        max_ted = self.count_nodes_in_tree(tree_gt)

        if max_ted == 0:
            # edge case: ground-truth is empty
            return 1.0 if ted_distance == 0 else 0.0

        accuracy = 1.0 - (ted_distance / max_ted)
        accuracy = max(0.0, accuracy)

        if debug:
            print(f"[ted_based_accuracy] TED = {ted_distance}, maxTED = {max_ted}, final_acc = {accuracy:.4f}")

        return accuracy
    
    def generate_json_from_model(self, example, model, processor, max_new_tokens=1024, debug=False):
        # 1. Pobieramy 'messages' z example
        message = example["messages"]
        
        # 2. Budujemy prompt
        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        if debug:
            print("[generate_json_from_model] Prompt:\n", prompt_str)

        # 3. Wyciągamy obrazy (jeśli występują)
        image_inputs, _ = process_vision_info(message)

        # 4. Przygotowujemy encodings dla modelu
        inputs = processor(
            text=[prompt_str],          # batch jednoelementowy
            images=[image_inputs],
            return_tensors="pt",
            padding=True
        )
        # przenosimy na urządzenie modelu (załóżmy CUDA)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 5. Generujemy predykcję
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=0.0,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # 6. Dekodujemy tokeny w string
        generated_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if debug:
            print("[generate_json_from_model] Generated text:\n", generated_text)

        return generated_text
    
    def evaluate_ted_accuracy_on_testset(self, test_set, model, processor, custom_metrics, debug=False):
        """
        Dla każdego przykładu w test_set:
        - odczytuje ground-truth JSON (z pliku),
        - generuje JSON z modelu,
        - liczy ted_based_accuracy,
        - zwraca średnią ze wszystkich przykładów.
        """
        test_accuracies = []

        for idx, example in enumerate(test_set):
            # 1) Wczytujemy ground-truth JSON w formie stringu
            json_gt_path = example["json_ground_path"]
            try:
                with open(json_gt_path, "r", encoding="utf-8") as f:
                    gt_json_str = f.read()
            except Exception as e:
                print(f"[WARN] Błąd wczytywania ground-truth: {json_gt_path} | {e}")
                # Możesz pominąć lub ustawić 0.0, w zależności od potrzeb
                continue

            # 2) Generujemy JSON z modelu
            pred_json_str = self.generate_json_from_model(example, model, processor, debug=debug)

            # 3) Liczymy TED-based Accuracy
            ted_acc = self.ted_based_accuracy(
                json_pred_str=pred_json_str, 
                json_gt_str=gt_json_str,
                debug=debug
            )
            test_accuracies.append(ted_acc)

            if debug:
                print(f"[evaluate_ted_accuracy_on_testset] idx={idx}, ted_acc={ted_acc:.4f}")

        final_acc = sum(test_accuracies) / len(test_accuracies) if len(test_accuracies) > 0 else 0.0
        return final_acc