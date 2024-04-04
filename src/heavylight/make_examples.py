# creates a folder with examples

'''
usage:
```python
import heavylight
heavylight.make_example('example', 'path/to/folder')

or using a simple CLI

'''
import shutil
from pathlib import Path
from typing import Union, Literal

examples = ['template', 'protection']
ExampleName = Literal['template', 'protection']

def make_example(download_to_path: Union[str, Path], example_name: ExampleName):
    if not example_name in examples:
        raise ValueError(f'example {example_name} not in available examples {examples}')
    example_path = Path(__file__).resolve().parent / 'examples' / example_name
    dest_path = Path(download_to_path).resolve() / example_name
    print(f'Creating {example_name} at {dest_path}')
    shutil.copytree(src=example_path, dst=dest_path)