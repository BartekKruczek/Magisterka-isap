import json
import zss

class JsonTree:
    def __init__(self, data, label=''):
        self.label = label
        self.children = []

        if isinstance(data, dict):
            # Dla obiektów JSON (słowników), etykietą jest klucz, a wartość to poddrzewo
            for key, value in data.items():
                child = JsonTree(value, label=str(key))
                self.children.append(child)
        elif isinstance(data, list):
            # Dla list, etykietą może być indeks lub ogólna etykieta
            for idx, item in enumerate(data):
                child = JsonTree(item, label=str(idx))
                self.children.append(child)
        else:
            # Dla węzłów liści (wartości), etykietą jest sama wartość
            self.label = str(data)

    def get_children(self):
        return self.children

    def get_label(self):
        return self.label

def compute_ted(json_obj1, json_obj2):
    """
    Oblicza Odległość Edycyjną Drzewa między dwoma obiektami JSON.

    :param json_obj1: Pierwszy obiekt JSON (parsowany do struktur Pythona)
    :param json_obj2: Drugi obiekt JSON (parsowany do struktur Pythona)
    :return: Odległość Edycyjna Drzewa (liczba całkowita)
    """
    tree1 = JsonTree(json_obj1)
    tree2 = JsonTree(json_obj2)
    distance = zss.simple_distance(
        tree1,
        tree2,
        get_children=lambda node: node.get_children(),
        get_label=lambda node: node.get_label(),
        label_dist=lambda label1, label2: 0 if label1 == label2 else 1
    )
    return distance

# Przykładowe użycie
if __name__ == "__main__":
    json_str1 = '{"name": "Jan", "age": 30, "skills": ["Python", "Machine Learning"]}'
    json_str2 = '{"name": "Jan", "age": 31, "skills": ["Python", "Deep Learning"]}'

    json_obj1 = json.loads(json_str1)
    json_obj2 = json.loads(json_str2)

    ted_distance = compute_ted(json_obj1, json_obj2)
    print(f"Odległość Edycyjna Drzewa: {ted_distance}")
