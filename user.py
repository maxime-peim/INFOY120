import dataclasses as dc

@dc.dataclass
class User:
    id: int
    label: str
    dataset_name: str

    def __hash__(self):
        return hash((self.id, self.label, self.dataset_name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
        return self.id == other.id and self.label == other.label \
            and self.dataset_name == other.dataset_name