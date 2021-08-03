`npoclass` - Classify nonprofits using NTEE codes
---

### How to install
---

#### Manage environment

1. [Install Anaconda Python distribution](https://www.anaconda.com/products/individual)
2. Create and use an environment

    ```Python
    conda create --name py38 python=3.8 # Install an environment named py38, using Python 3.8 as backend.
    conda activate py38 # Activate the environment.
    pip3 install -r requirements.txt # Install required packages.
    ```
3. If you use Jupyter Notebook, you need to add the environment:

    ```Python
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=py38
    ```

#### "Install" the classifier as a function

After checking required packages using an environment, install `npoclass` is simple. It is wrapped as a function and can be imported with two lines:

```Python
import requests
exec(requests.get('https://raw.githubusercontent.com/ma-ji/npo_classifier/master/API/npoclass.py').text)
```

Then you will have a function: `npoclass(inputs, gpu_core=True, model_path=None, ntee_type='bc', n_jobs=4, backend='multiprocessing')`


### How to use
---

#### Input parameters:
- `inputs`: a string text or a list of strings. For example:
    - A string text: `'We protect environment.'`
    - A list of strings: `['We protect environment.', 'We protect human.', 'We support art.']`
- `gpu_core=True`: Use GPU as default if GPU core is available.
- `model_path='npoclass_model_bc/'`: Path to model and label encoder files. Can be downloaded [here](https://jima.me/open/npoclass_model_bc.zip) (387MB).
- `ntee_type='bc'`: Predict broad category ('bc') or major group ('mg').
- `n_jobs=4`: The number of workers used to encode text strings.
- `backend='multiprocessing'`: Be one of {`'multiprocessing'`, `'sequential'`, `'dask'`}. Define the backend for parallel text encoding.
    - `multiprocessing`: Use `joblib`'s [`multiprocessing`](https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation) backend.
    - `sequential`: No parallel encoding and `n_jobs` ignored.
    - `dask`: Use [`dask.distributed`](https://distributed.dask.org/en/latest/client.html) as backend. `n_jobs` ignored and use all cluster workers. Follow [this post](https://jima.me/?p=950) for detail instruction.

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


### Suggestions on efficient computing
---

Two steps of prediction are time-consuming: 1) encoding raw text as vectors and 2) predicting classes using the model. Running Step 2 on GPU is much faster than on CPU and can hardly be optimized unless you have multiple GPUs. If you have a large amount of long text documents (e.g., several thousands of documents, and each has a thousand words), Step 1 will be very time-consuming if you go `sequential`. `dask` is only recommended if you have a huge amount of data; otherwise, `multiprocessing` is good enough because the scheduling step in cluster-computing also eats time. Which one works for you? You decide!


<!-- ### TODOs:
- List of Q&A.
    - [x] Use GPU or CPU.
    - <s> OMM errors.</s>
- [x] Parallel input encoding.
- <s>Publish on PyPI.</s> -->