from heavylight import make_example
from pathlib import Path
import shutil
import pytest

def test_make_example_bad():
    with pytest.raises(ValueError):
        make_example('temp', 'bad')

def test_make_example_protection():
    make_example('temp', 'protection')
    assert (Path('temp') / 'protection').exists()
    shutil.rmtree('temp/protection')

