`npoclass` - Classify nonprofits using NTEE codes
---

### How to install

Install `npoclass` is simple. It is wrapped as a function which can be imported with two lines:

```Python
import requests
exec(requests.get('https://raw.githubusercontent.com/ma-ji/npoclass.py').text)
```

Then you will have a function: `npoclass(inputs, gpu_core=True, model_path='npoclass_model/', ntee_type='bc')`

#### Input parameters:
- `inputs`: a string text or a list of strings. For example:
    - A string text: `'We protect environment.'`
    - A list of strings: `['We protect environment.', 'We protect human.', 'We support art.']`
- `gpu_core=True`: Use GPU as default if GPU core is available.
- `model_path='npoclass_model/'`: Path to model and label encoder files. Can be downloaded [here](https://jima.me/open/npoclass_model.zip) (387MB).
- `ntee_type='bc'`: Choose to predict broad category ('bc') or major group ('mg').

<!-- #### Output results:
- If input is a string:
- If input is a list of strings: -->


### TODOs:
- List of Q&A.
    - [x] Use GPU or CPU.
    - <s> OMM errors.</s>
- [ ] Workflow.
- [ ] Parallel input encoding.
- <s>Publish on PyPI.</s>