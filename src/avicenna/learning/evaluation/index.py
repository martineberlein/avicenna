from isla.derivation_tree import DerivationTree


class Index:
    def __init__(self, root: DerivationTree):
        self.root = root
        self.indexes = {}
        # Index nodes starting from the root
        self.index_nodes(self.root)

    def index_nodes(self, node: DerivationTree):
        # Use node IDs as keys to store indexes per node
        if node.id in self.indexes:
            return  # Already indexed
        index = {}

        def action(_, n):
            if n.value not in index:
                index[n.value] = []
            index[n.value].append(n)

        node.traverse(action)
        self.indexes[node.id] = index

    def get_nodes(self, node: DerivationTree, non_terminal: str) -> list[DerivationTree]:
        # Ensure the node's subtree is indexed
        self.index_nodes(node)
        index = self.indexes[node.id]
        return index.get(non_terminal, [])

