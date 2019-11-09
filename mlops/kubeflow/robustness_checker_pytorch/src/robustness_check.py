# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import argparse
import os

from robustness import robustness_check

def get_secret(path, default=''):
    try:
        with open(path, 'r') as f:
            cred = f.readline().strip('\'')
        f.close()
        return cred
    except:
        return default

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, help='Epsilon value for the FGSM attack', default=0.2)
    parser.add_argument('--model_id', type=str, help='Training model id', default="training-dummy")
    parser.add_argument('--metric_path', type=str, help='Path for robustness check output', default="/tmp/robustness.txt")
    parser.add_argument('--robust_status', type=str, help='Path for robustness status output', default="/tmp/status.txt")
    parser.add_argument('--model_class_file', type=str, help='pytorch model class file', default="model.py")
    parser.add_argument('--model_class_name', type=str, help='pytorch model class name', default="model")
    parser.add_argument('--loss_fn', type=str, help='PyTorch model loss function', default="torch.nn.CrossEntropyLoss()")
    parser.add_argument('--optimizer', type=str, help='pytorch model optimizer', default="torch.optim.Adam(model.parameters(), lr=0.001)")
    parser.add_argument('--clip_values', type=str, help='pytorch model clip_values allowed for features (min, max)', default="(0,1)")
    parser.add_argument('--nb_classes', type=int, help='The number of classes of the model', default=2)
    parser.add_argument('--input_shape', type=str, help='The shape of one input instance for the pytorch model', default="(1,3,64,64)")
    parser.add_argument('--feature_testset_path', type=str, help='Feature test dataset path in the data bucket', default="processed_data/X_test.npy")
    parser.add_argument('--label_testset_path', type=str, help='Label test dataset path in the data bucket', default="processed_data/y_test.npy")
    parser.add_argument('--data_bucket_name', type=str, help='Bucket that has the processed data', default="training-data")
    parser.add_argument('--result_bucket_name', type=str, help='Bucket that has the training results', default="training-result")
    args = parser.parse_args()

    epsilon = args.epsilon
    metric_path = args.metric_path
    model_id = args.model_id
    robust_status = args.robust_status
    model_class_file = args.model_class_file
    model_class_name = args.model_class_name
    LossFn = args.loss_fn
    Optimizer = args.optimizer
    nb_classes = args.nb_classes
    feature_testset_path = args.feature_testset_path
    label_testset_path = args.label_testset_path
    data_bucket_name = args.data_bucket_name
    result_bucket_name = args.result_bucket_name
    clip_values = eval(args.clip_values)
    input_shape = eval(args.input_shape)

    object_storage_url = get_secret('/app/secrets/s3_url', 'minio-service:9000')
    object_storage_username = get_secret('/app/secrets/s3_access_key_id', 'minio')
    object_storage_password = get_secret('/app/secrets/s3_secret_access_key', 'minio123')

    metrics = robustness_check(object_storage_url, object_storage_username, object_storage_password,
                               data_bucket_name, result_bucket_name, model_id,
                               feature_testset_path=feature_testset_path,
                               label_testset_path=label_testset_path,
                               clip_values=clip_values,
                               nb_classes=nb_classes,
                               input_shape=input_shape,
                               model_class_file=model_class_file,
                               model_class_name=model_class_name,
                               LossFn=LossFn,
                               Optimizer=Optimizer,
                               epsilon=epsilon)

    if not os.path.exists(os.path.dirname(metric_path)):
        os.makedirs(os.path.dirname(metric_path))
    with open(metric_path, "w") as report:
        report.write(json.dumps(metrics))

    robust = "true"
    if metrics['model accuracy on adversarial samples'] < 0.2:
        robust = "false"

    if not os.path.exists(os.path.dirname(robust_status)):
        os.makedirs(os.path.dirname(robust_status))
    with open(robust_status, "w") as report:
        report.write(robust)
