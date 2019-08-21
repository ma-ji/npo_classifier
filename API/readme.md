`npoclass` - Classify nonprofits into NTEE categories
---

### How to install

Install `npoclass` is simple. It is actually wrapped as a function. You can import it with two lines:

```Python
import requests
exec(requests.get('https://raw.githubusercontent.com/ma-ji/***.py').text)
```

Then you will have a function: `npoclass(string_input=None)`
- Input ( `string_input` ): a string text or a list of strings. For example:
    - A string text: 'We protect environment.'
    - A list of strings: ['We protect environment.', 'We protect human.', 'We support art.']
- Output: A Python dictionary is returned with the following data:
    - `major_group_label`: a list of predicted major group labels excluding "unknown" (i.e., A-Y).
    - `major_group_prob`: Probabilities of predictions. The higher the better.
    - `broad_category_label`: a list of predicted broad category labels excluding "unknown" (i.e., I-IX).
    - `broad_category_prob`: Probabilities of predictions. The higher the better.

### TODOs:
- [ ] List of Q&A.
    - OMM errors.
    - Use GPU or CPU.
- [ ] Workflow.
- [ ] Publish as a package to aviod importing unnessary classes.