import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('auc',Path(__file__).parents[1]/'analysis'/'auc.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def test_normalized_auc_constant_curve():
    assert np.isclose(mod.normalized_auc([100,500],[2,2],1000),2.0)
