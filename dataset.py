import json
import pathlib
from collections import namedtuple

import torch
from torchvision.io import read_image as tv_read_image
from torchvision import transforms

from PIL import Image

# def read_image(path):
#     return tv_read_image(str(path))
def read_image(path):
    return transforms.ToTensor()(Image.open(str(path)))

Example = namedtuple('Example', ['input', 'options', 'solution'])
Problem = namedtuple('Problem', ['examples', 'query', 'query_options', 'solution', 'program'])

class AnalogicalReasoningDataset(torch.utils.data.Dataset):
    """Analogical Reasoning Classification Dataset

    This dataset consists of analogical reasoning [1] classification problems.
    Each problem is made up of 5 parts
        1. Examples: a list of examples of a program where each example 
        is a tuple made of -
            i. Input image
            ii. Options for the output image among which one is the correct answer
            iii. The index of the correct option 
        2. Query: the query image for which to predict the correct option
        when the program (which should be inferred from the examples) is applied
        3. Query Options: the options for the output when the program inferred
        from the the examples is applied to the query
        4. Solution: the index of the correct option 
        5. Program: the ground truth program that generated the query image
    
    Calling `__getitem__` on this dataset returns a tuple containing 
        1. The problem with symbolic descriptions of the images
        2. The problem with images themselves
    Each of the above is a namedtuple with the following fields:
        1. `examples`: a list of examples of a program where each example
        is a tuple made of `input`, `options`, and `solution`.
        2. `query`
        3. `query_options`
        4. `solution`
        5. `program`
    Since this is a named tuple you can access these fields using their names.
    For example `problem.examples[0].input`

    This can be considered as a meta learning [2] task where each task consits 
    of training data (examples) and test data (query). You can split this dataset
    into meta-training and meta-test subsets. The meta-training subset can be
    used to train your model to be able to solve the query task given the examples.
    The meta-test subset can be used to test your model's performance.

    Args:
        problems_file: path to a json file containing the problems
        image_dir: path to the directory containing the images
        size (int): size of the dataset (should be given in the name of image_dir)

    References:
        [1] https://arxiv.org/abs/2111.10361
        [2] https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html#define-the-meta-learning-problem
    """
    def __init__(self, problems_file, image_dir, size1,size2):
        self.problems_file = problems_file
        self.image_dir = pathlib.Path(image_dir)
        self.size = size2-size1

        with open(self.problems_file, "r") as f:
            self.problems = json.load(f)[size1:size2]
        
    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]

        return Problem._make((
            [Example._make(e) for e in problem[0]],
            problem[1],
            problem[2],
            problem[3],
            problem[4],
        )), Problem._make((
            [
                Example._make((
                    read_image(self.image_dir/f"{idx}_{eidx}_input.png"),
                    [
                        read_image(self.image_dir/f"{idx}_{eidx}_{oidx}_option.png")
                        for oidx in range(len(problem[0][eidx][1]))
                    ],
                    problem[0][eidx][2]
                ))
                for eidx in range(len(problem[0]))
            ],
            read_image(self.image_dir/f"{idx}_query.png"),
            [
                read_image(self.image_dir/f"{idx}_{oidx}_query_option.png")
                for oidx in range(len(problem[2]))
            ],
            problem[3],
            problem[4]
        ))
