# ReadMe for Chloe

## getting start
```shell
source venv/bin/activate
# with docker image
make build # only run once; if you change ART library, you need to build again
make run-jupyter
# without docker image
pip3 install -r requirements.txt
pip install adversarial-robustness-toolbox
python examples/get_started_pytorch.py
# after you make change to art class
pip install .
```
## VS code tips
``` text
# On Mac:
`Ctrl + - ` navigate back
`Ctrl + Shift + -` navigate forward
```
## Open Jyupter Notebook on local

In browser
enter url that shows in the terminal where you run `make run-jupyter`. i.e. http://127.0.0.1:8888/?token=377c9499be5ae269ab26c3cda6f54e01af14fe679718fedd, please note the token changes everytime

## Q&A
**Question**: Error "No such file or directory: 'git'"
```shell
Collecting git+https://github.com/nottombrown/imagenet_stubs
  Cloning https://github.com/nottombrown/imagenet_stubs to /tmp/pip-req-build-an_vftp0
  Running command git clone -q https://github.com/nottombrown/imagenet_stubs /tmp/pip-req-build-an_vftp0
  ERROR: Error [Errno 2] No such file or directory: 'git': 'git' while executing command git clone -q https://github.com/nottombrown/imagenet_stubs /tmp/pip-req-build-an_vftp0
ERROR: Cannot find command 'git' - do you have 'git' installed and in your PATH?
```
**Answer**: run command in docker container
```shell
apt-get install git
```

----
**Question**: Error "AlreadyExistsError: Another metric with the same name already exists."
**Answer**: pip install following commands
```shell
# supported versions: (tensorflow==2.2.0 with keras==2.3.1) or (tensorflow==1.15.4 with keras==2.2.5)
pip install tensorflow==2.2.0
pip show tensorflow
pip install keras==2.3.1
pip show keras
```
----
**Question**: Error "ValueError: The shape of mean and the standard deviation must be identical."
**Answer**: to be added

---

**Question**: Error "cannot import name 'SignOPTAttack' from 'art.attacks.evasion'"
```shell
python examples/chloe-pytorch.py 
Traceback (most recent call last):
  File "examples/chloe-pytorch.py", line 13, in <module>
    from art.attacks.evasion import FastGradientMethod, BoundaryAttack, SignOPTAttack
ImportError: cannot import name 'SignOPTAttack' from 'art.attacks.evasion' (/Users/chloe/git/trusted-ai/adversarial-robustness-toolbox/venv/lib/python3.8/site-packages/art/attacks/evasion/__init__.py)
```
**Answer**: since you made change to ART package, need to install ART again
```shell
# after you make change to art class
pip install .
```
---

**Question** Error "4-dimensional input for 4-dimensional weight [4, 1, 5, 5], but got 3-dimensional input of size [1, 28, 28] instead"
```shell
Exception has occurred: RuntimeError
Expected 4-dimensional input for 4-dimensional weight [4, 1, 5, 5], but got 3-dimensional input of size [1, 28, 28] instead
  File "/Users/chloe/git/trusted-ai/adversarial-robustness-toolbox/art/attacks/evasion/test-by-chloe.py", line 30, in forward
    x = F.relu(self.conv_1(x))
  File "/Users/chloe/git/trusted-ai/adversarial-robustness-toolbox/art/attacks/evasion/test-by-chloe.py", line 90, in <module>
    y0 = classifier.predict(x_test[0])
```
**Answer** PyTorchClassifier's predict() method in ART is used for batch of dataset prediction, for one data sample prediction, you can use predict() by creating a mini-batch of size 1, the shape would be (1, height, width, channels); Or extend the data to be 
```python
np.expand_dims(x0, axis=0)
```
the dimension is changed from [1, 28, 28] to [1, 1, 28, 28]

---
**Question**
```shell
Traceback (most recent call last):
  File "examples/get_started_pytorch.py", line 81, in <module>
    x_test_adv = attack.generate(x=x_test[:5])
  File "/Users/chloe/git/trusted-ai/adversarial-robustness-toolbox/venv/lib/python3.8/site-packages/art/attacks/evasion/sign_opt.py", line 102, in generate
    100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
AttributeError: 'SignOPTAttack' object has no attribute 'batch_size'
```
**Answer**
todo

---
**Question**
Errors when collecting PyTest tests
**Answer**
1. installed several package according to output
2. `brew install libomp` for error "XGBoost Library (libxgboost.dylib) could not be loaded. "

# Notes
- pandas, create DataFrame, insert a column, add a row
```python
# create an empty DataFrame, init a column 
df = pandas.DataFrame({'l2': []}) 
# add a row
df.loc[0] = 1.5
# insert a column 
df.insert(1, "newcol", [])
```

- install python package from certain github branch
```shell
pip install git+https://github.com/[repo owner]/[repo]@[branch name]
# take my branch for example
pip install git+https://github.com/synergit/adversarial-robustness-toolbox@development_issue_1331
```

Q:
4th and 9th images in back_end_untargeted_images() make sign_OPT() can't find a good starting direction, why? 