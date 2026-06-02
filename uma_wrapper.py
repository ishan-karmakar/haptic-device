from ase import Atoms
from calculator import _resolve_calculator


def create_calculator(spec):
    return _resolve_calculator(spec)