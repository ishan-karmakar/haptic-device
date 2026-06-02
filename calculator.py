from importlib import import_module
from ast import literal_eval

from ase import Atoms

from fairchem.core import pretrained_mlip, FAIRChemCalculator


_uma_predictor_cache = {}


def _get_uma_predictor(model_name="uma-s-1p2", device="cuda"):
    key = (model_name, device)

    if key not in _uma_predictor_cache:
        _uma_predictor_cache[key] = pretrained_mlip.get_predict_unit(
            model_name,
            device=device,
        )

    return _uma_predictor_cache[key]


def _resolve_calculator(spec):

    if not spec or spec in {"lj", "lennard-jones"}:
        module_name = "ase.calculators.lj"
        class_name = "LennardJones"
        kwargs = {}

        calculator_class = getattr(import_module(module_name), class_name)
        return calculator_class(**kwargs)

    elif spec == "morse":
        module_name = "ase.calculators.morse"
        class_name = "MorsePotential"
        kwargs = {}

        calculator_class = getattr(import_module(module_name), class_name)
        return calculator_class(**kwargs)

    elif spec == "emt":
        module_name = "ase.calculators.emt"
        class_name = "EMT"
        kwargs = {}

        calculator_class = getattr(import_module(module_name), class_name)
        return calculator_class(**kwargs)

    elif spec.startswith("uma"):

        # Examples:
        # "uma"
        # "uma:omol"
        # "uma:omat"
        # "uma:oc20"

        parts = spec.split(":")

        task_name = "omol"

        if len(parts) > 1:
            task_name = parts[1]

        predictor = _get_uma_predictor(
            model_name="uma-s-1p2",
            device="cuda"
        )

        return FAIRChemCalculator(
            predictor,
            task_name=task_name
        )

    else:
        parts = spec.split(":", 2)

        if len(parts) < 2:
            raise ValueError(
                "Calculator spec must be empty, a known alias, or module:Class[:kwargs]"
            )

        module_name, class_name = parts[0], parts[1]

        kwargs = literal_eval(parts[2]) if len(parts) == 3 else {}

        if not isinstance(kwargs, dict):
            raise ValueError("Calculator kwargs must evaluate to a dict")

        calculator_class = getattr(import_module(module_name), class_name)

        return calculator_class(**kwargs)


def get_values(numbers, positions, cell=None, pbc=None, calculator_spec=""):

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc
    )

    # Required for UMA omol task
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1

    atoms.calc = _resolve_calculator(calculator_spec)

    return {
        "forces": atoms.get_forces().tolist(),
        "energy": atoms.get_potential_energy(),
    }


if __name__ == "__main__":

    sample = get_values(
        numbers=[1, 1],
        positions=[[0, 0, 0], [1.0, 0, 0]],
        calculator_spec="uma:omol"
    )

    print(sample)