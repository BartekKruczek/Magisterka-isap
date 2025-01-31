import json
import zss

from json_handler import JsonHandler

my_json_handler = JsonHandler()

json_path1 = "lemkin-json-from-html/1918/1918_2.json"
json_path2 = "lemkin-json-from-html/1918/1918_17.json"

string1: str = ""
with open(json_path1, 'r', encoding = 'utf-8') as file1:
    string1: str = file1.read()

json1: dict = json.loads(string1)

string2: str = ""
with open(json_path2, 'r', encoding = 'utf-8') as file2:
    string2: str = file2.read()

json2: dict = json.loads(string2)

json1token = my_json_handler.json_to_token_conversion(json_obj = json1, keys_only = True)
json2token = my_json_handler.json_to_token_conversion(json_obj = json2, keys_only = True)

print(json1token)
print(json2token)

# tree from tokens
json1tree_nodes, _ = my_json_handler.create_tree_from_tokens(json1token)
json2tree_nodes, _ = my_json_handler.create_tree_from_tokens(json2token)

calc_dist = zss.simple_distance(
    A=json1tree_nodes,
    B=json2tree_nodes,
    get_children=lambda my_node: my_node.get_children(),
    get_label=lambda my_node: my_node.get_label(),
    label_dist=lambda label1, label2: 0 if label1 == label2 else 1
)

# Tworzenie drzewa bezpośrednio z obiektów JSON (bez tokenów)
json1tree_no_tokens = my_json_handler.create_tree_from_json_string(json=json1)
json2tree_no_tokens = my_json_handler.create_tree_from_json_string(json=json2)

# Obliczanie TED bez pośrednictwa tokenów
calc_dist_no_tokens = zss.simple_distance(
    A=json1tree_no_tokens,
    B=json2tree_no_tokens,
    get_children=lambda my_node: my_node.get_children(),
    get_label=lambda my_node: my_node.get_label(),
    label_dist=lambda label1, label2: 0 if label1 == label2 else 1
)

print("TED (bezpośrednio z JSON):", calc_dist_no_tokens)
print("TED:", calc_dist)