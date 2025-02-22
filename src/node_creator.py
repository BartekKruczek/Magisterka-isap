class CustomNodeCreator:
    def __init__(self, label, children = None) -> None:
        self.label: str = label
        
        if children is not None:
            self.children = children
        else:
            self.children: list = []

    def __repr__(self) -> str:
        return "Klasa do tworzenia node-ów z plików json, aby obliczyć metrykę TED"
    
    def get_children(self) -> list:
        return self.children
    
    def get_label(self) -> str:
        return self.label
    
    def add_children(self, node):
        return self.children.append(node)