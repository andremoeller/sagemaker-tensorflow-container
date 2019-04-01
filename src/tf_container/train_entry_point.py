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

import argparse
import json
import os
import subprocess
import time

import tensorflow as tf
import psutil

import container_support as cs
import tf_container.run
import tf_container.serve as serve

from tf_container.trainer import Trainer
from multiprocessing import Process
from tf_container.timeout import timeout


_logger = tf_container.run.get_logger()

_PROCESS_ATTRIBUTES = ['name', 'pid', 'cpu_affinity', 'cpu_num', 'cpu_percent', 'cpu_times','io_counters',
                         'ionice','memory_full_info', 'memory_percent', 'nice','num_ctx_switches',
                         'num_threads','status', 'threads']


def _wait_until_master_is_up(master):
    with timeout(minutes=10):
        while True:
            try:
                # this subprocess call is python 2/3 compatible and will throw an exception when the status code is != 0
                subprocess.check_call(['curl', '{}:2222'.format(master)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                _logger.info("master node is up.")
                return
            except subprocess.CalledProcessError:
                _logger.info("master node is not up yet")
                time.sleep(10)


def _wait_until_master_is_down(master, psutil_processes):
    while True:
        try:
            # this subprocess call is python 2/3 compatible and will throw an exception when the status code is != 0
            subprocess.check_call(['curl', '{}:2222'.format(master)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for p in psutil_processes:
                _logger.info('process information: {}'.format(p.as_dict(attrs=_PROCESS_ATTRIBUTES)))
            time.sleep(10)
        except subprocess.CalledProcessError:
            _logger.info("master {} is down, exiting".format(master))
            return


def save_tf_config_env_var(tf_config):
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    _logger.info('----------------------TF_CONFIG--------------------------')
    _logger.info(os.environ['TF_CONFIG'])
    _logger.info('---------------------------------------------------------')


def _run_ps_server(current_host, hosts, tf_config):
    """After the training finishes, parameter servers won't stop running because server.join() has an infinite loop.
    That is a known issue: https://github.com/tensorflow/ecosystem/issues/19
    The solution below, runs the parameter server in a secondary thread while the main thread pings the master waiting
    for it to stop responding. After that, it will exit the application gracefully given that python threads cannot be
    stopped

    Args:
        current_host: (str) name of the current host
        hosts: list (str) list of all the hostnames
        tf_config: dict (str) tensorflow config map

    Returns:
    """

    def start_ps_server(current_host, hosts, tf_config):
        cluster_spec = tf.train.ClusterSpec(tf_config['cluster'])
        task_index = hosts.index(current_host)
        server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index)
        server.join()

    p = Process(name='parameter-server', target=start_ps_server, args=(current_host, hosts, tf_config))
    _logger.info('Starting parameter server process')
    p.start()
    psutil_process = psutil.Process(p.pid)
    # One physical CPU for PS.
    # TODO: tune this. Small instances, greater throughput.
    #psutil_process.cpu_affinity([0, 1])
    return psutil_process


def _run_workers(current_host, hosts, tf_config, hyperparameters, trainer):
    workers_per_host = hyperparameters.get('workers_per_host', 1)

    def start_worker(current_host, hosts, tf_config, worker_index):
        # This assumes we're running n - 1 worker tasks, and that worker_index >= 1.
        task_index = (hosts.index(current_host) + (len(hosts) - 1) * worker_index) - 1
        _logger.info('starting worker process with index {} on node, and task index {}'.format(worker_index, task_index))
        tf_config['task'] = {
                'index': task_index,
                'type': 'worker'
        }
        save_tf_config_env_var(tf_config)
        trainer.train()
        _wait_for_master_node(tf_config, trainer)

    def _partition(lst, n):
        division = len(lst) / float(n)
        return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]

    def physical_cpus_to_logical_cpus(l):
        return sorted([item * 2 for item in l] + [item * 2 + 1 for item in l])
    # PS has logical CPUs 0, 1 (or physical CPU 1).
    # Count physical CPUs rather than logical CPUs to ensure that physical CPUs aren't split between processes.
    cpu_list = list(range(psutil.cpu_count(False)))[1:]
    psutil_processes = []
    partitioned_cpu_list = [physical_cpus_to_logical_cpus(part) for part in _partition(cpu_list, workers_per_host)]
    for worker_index in range(workers_per_host):
        if worker_index == 0:
            # This worker was already started by TF session
            psutil_process = psutil.Process()
        else:
            p = Process(name='worker-{}'.format(worker_index), target=start_worker,
                        args=(current_host, hosts, tf_config, worker_index))
            p.start()
            psutil_process = psutil.Process(p.pid)
        affinity = partitioned_cpu_list[worker_index]
        #_logger.info('Setting worker index {} to have CPU affinity {}'.format(worker_index, affinity))
        #psutil_process.cpu_affinity(affinity)
        psutil_processes.append(psutil_process)
    return psutil_processes




def _get_default_training_params(env):
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--training_steps', type=int, default=1000)
    my_parser.add_argument('--evaluation_steps', type=int, default=100)
    hp = env.argparse_hyperparameters(my_parser)

    return hp.training_steps, hp.evaluation_steps


def _get_master(tf_config):
    return tf_config['cluster']['master'][0][:-5]


def _get_checkpoint_dir(env):
    if 'checkpoint_path' not in env.hyperparameters:
        return env.model_dir

    checkpoint_path = env.hyperparameters['checkpoint_path']

    # If this is not part of a tuning job, then we can just use the specified checkpoint path
    if '_tuning_objective_metric' not in env.hyperparameters:
        return checkpoint_path

    job_name = env.job_name

    # If the checkpoint path already matches the format 'job_name/checkpoints', then we don't
    # need to worry about checkpoints from multiple training jobs being saved in the same location
    if job_name is None or checkpoint_path.endswith(os.path.join(job_name, 'checkpoints')):
        return checkpoint_path
    else:
        return os.path.join(checkpoint_path, job_name, 'checkpoints')


def configure_mkl():
    """Sets MKL env variables with default settings.
    More information about how to setup MLK: ENV KMP_AFFINITY= KMP_BLOCKTIME=1 KMP_SETTINGS=0"""
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['KMP_SETTINGS'] = '0'


def train():
    env = cs.TrainingEnvironment()

    checkpoint_dir = _get_checkpoint_dir(env)
    train_steps = env.hyperparameters.get('training_steps', 1000)
    eval_steps = env.hyperparameters.get('evaluation_steps', 100)

    # https://github.com/tensorflow/tensorflow/issues/15868
    # The default request timeout for S3, within the C++ SDK, is 3 seconds, which times out when
    # saving checkpoints of larger sizes.
    os.environ['S3_REQUEST_TIMEOUT_MSEC'] = str(env.hyperparameters.get('s3_checkpoint_save_timeout', 60000))

    if env.user_script_archive.lower().startswith('s3://'):
        env.download_user_module()
    env.pip_install_requirements()

    customer_script = env.import_user_module()

    _logger.info('{} physical cores, {} logical cores'.format(psutil.cpu_count(False), psutil.cpu_count(True)))
    _logger.info('psutil cpu freq: {}'.format(psutil.cpu_freq(percpu=True)))

    trainer = Trainer(customer_script=customer_script,
                            current_host=env.current_host,
                            hosts=env.hosts,
                            train_steps=train_steps,
                            eval_steps=eval_steps,
                            input_channels=env.channel_dirs,
                            model_path=checkpoint_dir,
                            output_path=env.output_dir,
                            customer_params=env.hyperparameters)

    tf_config = trainer.build_tf_config()
    save_tf_config_env_var(tf_config)

    processes = []
    if len(env.hosts) > 1:
        parameter_server_process = _run_ps_server(env.current_host, env.hosts, tf_config)
        processes.append(parameter_server_process)
        if env.current_host != 'algo-1':
            worker_processes = _run_workers(env.current_host, env.hosts, tf_config, env.hyperparameters, trainer)
            processes += worker_processes

    configure_mkl()

    trainer.train()

    _logger.info('psutil cpu stats: {}'.format(psutil.cpu_stats()))

    subprocess.check_output("ps aux", shell=True)

    # only the master should export the model at the end of the execution
    if checkpoint_dir != env.model_dir and trainer.task_type == 'master' and trainer.saves_training():
        serve.export_saved_model(checkpoint_dir, env.model_dir)

    if trainer.task_type != 'master':
        _wait_for_master_node(tf_config, trainer, processes)

def _wait_for_master_node(tf_config, trainer, processes=[]):
    _wait_until_master_is_up(_get_master(tf_config))
    _logger.info('task_type is {}, waiting for master to exit'.format(trainer.task_type))
    _wait_until_master_is_down(_get_master(tf_config), processes)