# creates a folder with examples

'''
usage:
```python
import heavylight
heavylight.make_example('example', 'path/to/folder')

or using a simple CLI

'''
import os
import shutil
import pathlib

examples = ['template', 'protection']

def make_example(path: str|pathlib.Path, example_name:str):
    if example_name in examples:
        base_src_path = pathlib.Path(os.path.realpath(__file__))
        # print(f"{base_src_path=}")
        example_path = base_src_path.parent / 'examples' / example_name
        # print(f"{example_path=}")
        dest_path = pathlib.Path(path) / example_name
        print(f'Creating {example_name} at {dest_path}')
        shutil.copytree(src=example_path, dst=dest_path)
        # TODO: print a nice view of the files and folders set up (like hatch new)

    else:
        raise ValueError(f'example {example_name} not in available examples {examples}')

if __name__ == '__main__':
    make_example('/Users/lewisfogden/Dev/pyscratch/temp', 'protection')