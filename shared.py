from registry import ModelRegistry
from data_storage import DataCatalog

registry = ModelRegistry()
catalog = DataCatalog()

# Import models to trigger registration decorators
import models