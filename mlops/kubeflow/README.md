# Kubeflow Pipeline Components for ART

Kubeflow pipeline components are implementations of Kubeflow pipeline tasks. Each task takes
one or more [artifacts](https://www.kubeflow.org/docs/pipelines/overview/concepts/output-artifact/)
as input and may produce one or more
[artifacts](https://www.kubeflow.org/docs/pipelines/overview/concepts/output-artifact/) as output.


**Example: ART Components**
* [Adversarial Robustness Evaluation - FGSM - PyTorch](robustness_evaluation_fgsm_pytorch)

Each task usually includes two parts:

Each component has a component.yaml which will describe the functionality exposed by it, for e.g.

```
name: 'PyTorch - Fast Gradient Sign Method - Adversarial Robustness Evaluation'
description: |
  Perform an evaluation of the adversarial robustness using the Fast Gradient Sign Method of the Adversarial Robustness Toolbox (ART).
metadata:
  annotations: {platform: 'OpenSource'}
inputs:
  - {name: model_id,                     description: 'Required. Training model ID', default: 'training-dummy'}
  - {name: epsilon,                      description: 'Required. Epsilon value for the FGSM attack'}
  - {name: model_class_file,             description: 'Required. pytorch model class file'}
  - {name: model_class_name,             description: 'Required. pytorch model class name', default: 'model'}
  - {name: feature_testset_path,         description: 'Required. Feature test dataset path in the data bucket'}
  - {name: label_testset_path,           description: 'Required. Label test dataset path in the data bucket'}
  - {name: loss_fn,                      description: 'Required. PyTorch model loss function'}
  - {name: optimizer,                    description: 'Required. pytorch model optimizer'}
  - {name: clip_values,                  description: 'Required. pytorch model clip_values allowed for features (min, max)'}
  - {name: nb_classes,                   description: 'Required. The number of classes of the model'}
  - {name: input_shape,                  description: 'Required. The shape of one input instance for the pytorch model'}
  - {name: data_bucket_name,             description: 'Bucket that has the processed data',  default: 'training-data'}
  - {name: result_bucket_name,           description: 'Bucket that has the training results', default: 'training-result'}
outputs:
  - {name: metric_path,                  description: 'Path for robustness check output'}
  - {name: robust_status,                description: 'Path for robustness status output'}
implementation:
  container:
    image: aipipeline/robustness-evaluation:pytorch
    command: ['python']
    args: [
      -u, robustness_evaluation_fgsm_pytorch.py,
      --model_id, {inputValue: model_id},
      --model_class_file, {inputValue: model_class_file},
      --model_class_name, {inputValue: model_class_name},
      --feature_testset_path, {inputValue: feature_testset_path},
      --label_testset_path, {inputValue: label_testset_path},
      --epsilon, {inputValue: epsilon},
      --loss_fn, {inputValue: loss_fn},
      --optimizer, {inputValue: optimizer},
      --clip_values, {inputValue: clip_values},
      --nb_classes, {inputValue: nb_classes},
      --input_shape, {inputValue: input_shape},
      --metric_path, {outputPath: metric_path},
      --robust_status, {outputPath: robust_status},
      --data_bucket_name, {inputValue: data_bucket_name},
      --result_bucket_name, {inputValue: result_bucket_name}
    ]
```

See how to [use the Kubeflow Pipelines SDK](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/)
and [build your own components](https://www.kubeflow.org/docs/pipelines/sdk/build-component/).
