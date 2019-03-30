#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import logging

import numpy as np
import pytest
from sagemaker.tensorflow import TensorFlow

from test.resources.python_sdk.timeout import timeout, timeout_and_delete_endpoint

logger = logging.getLogger(__name__)
logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)
logging.getLogger('session.py').setLevel(logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)

PIPE_MODE_VERSIONS = ['1.7.0', '1.8.0']


class MyEstimator(TensorFlow):
    def __init__(self, docker_image_uri, **kwargs):
        super(MyEstimator, self).__init__(**kwargs)
        self.docker_image_uri = docker_image_uri

    def train_image(self):
        return self.docker_image_uri

    def create_model(self, model_server_workers=None):
        model = super(MyEstimator, self).create_model()
        model.image = self.docker_image_uri
        return model


def test_distributed_one_worker(instance_type, sagemaker_session, docker_image_uri):
    script_path = 'test/resources/mnist/code'
    data_path = 'test/resources/mnist/data/training'
    workers_per_host = 1
    with timeout(minutes=30):
        estimator = MyEstimator(entry_point='mnist.py',
                                source_dir=script_path,
                                role='SageMakerRole',
                                training_steps=1000,
                                evaluation_steps=100,
                                base_job_name='test-distributed-{}-workers'.format(workers_per_host),
                                hyperparameters={'workers_per_host': workers_per_host},
                                train_instance_count=4,
                                train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=docker_image_uri)

        logger.info("uploading training data")
        key_prefix = 'integ-test-data/tf-cifar-{}'.format(instance_type)
        inputs = estimator.sagemaker_session.upload_data(path=data_path,
                                                         key_prefix=key_prefix)

        logger.info("fitting estimator")
        estimator.fit(inputs)


def test_distributed_four_workers(instance_type, sagemaker_session, docker_image_uri):
    script_path = 'test/resources/mnist/code'
    data_path = 'test/resources/mnist/data/training'
    workers_per_host = 4
    with timeout(minutes=30):
        estimator = MyEstimator(entry_point='mnist.py',
                                source_dir=script_path,
                                role='SageMakerRole',
                                training_steps=1000,
                                evaluation_steps=100,
                                base_job_name='test-distributed-{}-workers'.format(workers_per_host),
                                hyperparameters={'workers_per_host': workers_per_host},
                                train_instance_count=4,
                                train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=docker_image_uri)

        logger.info("uploading training data")
        key_prefix = 'integ-test-data/tf-cifar-{}'.format(instance_type)
        inputs = estimator.sagemaker_session.upload_data(path=data_path,
                                                         key_prefix=key_prefix)

        logger.info("fitting estimator")
        estimator.fit(inputs)


def test_distributed_two_workers(instance_type, sagemaker_session, docker_image_uri):
    script_path = 'test/resources/mnist/code'
    data_path = 'test/resources/mnist/data/training'
    workers_per_host = 2
    with timeout(minutes=30):
        estimator = MyEstimator(entry_point='mnist.py',
                                source_dir=script_path,
                                role='SageMakerRole',
                                training_steps=1000,
                                evaluation_steps=100,
                                base_job_name='test-distributed-{}-workers'.format(workers_per_host),
                                hyperparameters={'workers_per_host': workers_per_host},
                                train_instance_count=4,
                                train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=docker_image_uri)

        logger.info("uploading training data")
        key_prefix = 'integ-test-data/tf-cifar-{}'.format(instance_type)
        inputs = estimator.sagemaker_session.upload_data(path=data_path,
                                                         key_prefix=key_prefix)

        logger.info("fitting estimator")
        estimator.fit(inputs)


def test_pipe_mode(instance_type, sagemaker_session, docker_image_uri):
    framework_version = docker_image_uri.split(':')[-1].split('-')[0]
    if framework_version not in PIPE_MODE_VERSIONS:
        pytest.skip('skipping non-pipe-mode version {} because it is not in {}'
                    .format(framework_version, PIPE_MODE_VERSIONS))
    script_path = 'test/resources/synthetic'

    with timeout(minutes=15):
        estimator = MyEstimator(entry_point='synthetic_pipe_mode_dataset.py',
                                source_dir=script_path,
                                role='SageMakerRole',
                                input_mode='Pipe',
                                training_steps=100,
                                evaluation_steps=10,
                                train_instance_count=1,
                                train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=docker_image_uri)

        logger.info("uploading training data")

        train_data = 's3://sagemaker-sample-data-us-west-2/tensorflow/pipe-mode/train'
        eval_data = 's3://sagemaker-sample-data-us-west-2/tensorflow/pipe-mode/eval'

        logger.info("fitting estimator")
        estimator.fit({'train': train_data, 'eval': eval_data})
