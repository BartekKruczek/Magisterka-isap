import zss

from json_handler import JsonHandler

class CustomMetrics(JsonHandler):
    def __init__(self) -> None:
        super().__init__()

    def calculate_tree_edit_distance(self, json_generated_path: str = None, json_test_path: str = None, debug: bool = False) -> int:
        json_generated, json_test = self.json_load_TED(
            json_file1_path = json_generated_path, 
            json_file2_path = json_test_path,
            )

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