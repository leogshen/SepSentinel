# Model registry — factory for creating and discovering models.

_REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator that registers a SepsisModel subclass."""
    _REGISTRY[cls.name] = cls
    return cls


def get_model(name: str, **kwargs):
    """Create a model instance by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "(none registered)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return names of all registered models."""
    return list(_REGISTRY.keys())


# Import model modules to trigger registration.
# Each module uses @register on its model class.
import sepsentinel.models.random_forest  # noqa: E402, F401
