# Contributing to the Adversarial Robustness Toolbox

## Adding new Features
Adding new features, improving documentation, fixing bugs, or writing tutorials are all examples of helpful 
contributions. Furthermore, if you are publishing a new attack or defense, we strongly encourage you to add it to the 
Adversarial Robustness Toolbox so that others may evaluate it fairly in their own work.

Bug fixes can be initiated through GitHub pull requests. When making code contributions to the Adversarial Robustness 
Toolbox, we ask that you follow the `PEP 8` coding standard and that you provide unit tests for the new features.

Contributions of new features must include unit test covering at least 80% of the new statements.

## Validating Git Commits
This project uses [DCO](https://developercertificate.org/). Be sure to sign off your commits using the `-s` flag or 
adding `Signed-off-By: Name<Email>` in the commit message. Example:
```bash
git commit -s -m 'Informative commit message'
```

## Unit tests
When submitting additional unit tests for ART, in order to keep the code base maintainable, please make sure each unit 
test can run ideally in a few seconds.
