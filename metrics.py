import zss
import pandas as pd

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
    
    def calculate_TED_accuracy_on_set(dataset: pd.DataFrame, debug: bool = False) -> float:
        pass