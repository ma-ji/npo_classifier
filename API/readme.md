`npoclass` - Classify nonprofits using NTEE codes
---

### How to install

After checking package [`requirements.txt`](https://github.com/ma-ji/npo_classifier/blob/master/API/requirements.txt) (recommend using a [environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)), install `npoclass` is simple. It is wrapped as a function and can be imported with two lines:

```Python
import requests
exec(requests.get('https://raw.githubusercontent.com/ma-ji/npo_classifier/master/API/npoclass.py').text)
```

Then you will have a function: `npoclass(inputs, gpu_core=True, n_jobs=4, model_path='npoclass_model_bc/', ntee_type='bc')`

#### Input parameters:
- `inputs`: a string text or a list of strings. For example:
    - A string text: `'We protect environment.'`
    - A list of strings: `['We protect environment.', 'We protect human.', 'We support art.']`
- `gpu_core=True`: Use GPU as default if GPU core is available.
- `n_jobs=4`: The number of workers used to encode text strings.
- `model_path='npoclass_model_bc/'`: Path to model and label encoder files. Can be downloaded [here](https://jima.me/open/npoclass_model_bc.zip) (387MB).
- `ntee_type='bc'`: Predict broad category ('bc') or major group ('mg').

#### Output:

A list of result dictionaries in the order of the input. If the input is a string, the return list will only have one element. For example:

```Python
[{'recommended': 'II',
  'confidence': 'high (>=.99)',
  'probabilities': {'I': 0.5053213238716125,
   'II': 0.9996891021728516,
   'III': 0.752209484577179,
   'IV': 0.6053232550621033,
   'IX': 0.2062985599040985,
   'V': 0.9766567945480347,
   'VI': 0.27059799432754517,
   'VII': 0.8041080832481384,
   'VIII': 0.3203429579734802}},
 {'recommended': 'II',
  'confidence': 'high (>=.99)',
  'probabilities': {'I': 0.5053213238716125,
   'II': 0.9996891021728516,
   'III': 0.752209484577179,
   'IV': 0.6053232550621033,
   'IX': 0.2062985599040985,
   'V': 0.9766567945480347,
   'VI': 0.27059799432754517,
   'VII': 0.8041080832481384,
   'VIII': 0.3203429579734802}},
   ...,
]
```



<!-- ### TODOs:
- List of Q&A.
    - [x] Use GPU or CPU.
    - <s> OMM errors.</s>
- [x] Parallel input encoding.
- <s>Publish on PyPI.</s> -->