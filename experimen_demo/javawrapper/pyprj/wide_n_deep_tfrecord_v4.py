#-*-coding:utf-8-*-
"""
3层神经网络，然后把拼音的编辑距离去掉，APP的下载人数是one-hot为6位
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import json
import math
import six
import tensorflow as tf
from tensorflow.python.estimator.canned import linear
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.summary import summary
from tensorflow.python.training import training_util
import random


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)sline:%(lineno)d][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("run_mode", "local", "the run mode, one of local and distributed")
flags.DEFINE_string("job_type", "export_savedmodel", "the job type, one of train, eval, train_and_eval, export_savedmodel and predict")

flags.DEFINE_string("train_data", "E:\\ProductProject\\DNN\\datas\\part-r-00000", "the train data, delimited by comma")
flags.DEFINE_string("eval_data", "E:\\ProductProject\\DNN\\datas\\part-r-00000", "the eval data, delimited by comma")
flags.DEFINE_integer("batch_size", 1024, "the batch size")
flags.DEFINE_integer("feature_size", 10000, "the feature size")
flags.DEFINE_integer("num_epochs", None, "the epoch num")
flags.DEFINE_integer("train_steps", 100, "the train step")
flags.DEFINE_integer("eval_steps", 10, "the eval step")

flags.DEFINE_string("model_type", "wide_and_deep", "the model type, one of wide, deep and wide_and_deep")
flags.DEFINE_string("model_dir", "hdfs://footstone/data/project/dataming/browser_search_dev/fyang/wide_n_deep/model_version_dropout_0225", "the model dir")
flags.DEFINE_bool("cold_start", False, "True: cold start; False: start from the latest checkpoint")
flags.DEFINE_string("export_savedmodel", "hdfs://footstone/data/project/dataming/browser_search_dev/fyang/wide_n_deep/model_version_dropout_0225_out", "the export savedmodel directory, used for tf serving")
flags.DEFINE_string("savedmodel_mode", "parsing", "the savedmodel mode, one of raw and parsing")
flags.DEFINE_integer("eval_ckpt_id", 0, "the checkpoint id in model dir for evaluation, 0 is the latest checkpoint")
flags.DEFINE_string("file_format", "csv", "the file format, one of csv and tfrecord")

flags.DEFINE_string("worker_hosts", "localhost:12000,localhost:12001", "the worker hosts, delimited by comma")
flags.DEFINE_string("ps_hosts", "localhost:13001", "the ps hosts, delimited by comma")
flags.DEFINE_string("task_type", "worker", "the task type, one of worker and ps")
flags.DEFINE_integer("task_index", 0, "the task index, starting from 0")

_COLS_IDX_QUERY=0
_COLS_IDX_APPNAME=1

def _linear_logit_fn_builder(units, feature_columns):
  def linear_logit_fn(features):
    return feature_column_lib.linear_model(
        features=features, feature_columns=feature_columns, units=units)
  return linear_logit_fn


def _add_hidden_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)

def _dnn_logit_fn_builder(units, hidden_units, feature_columns, activation_fn,
                          dropout, input_layer_partitioner):
  if not (isinstance(units, int) or isinstance(units, list)):
    raise ValueError('units must be an int or list.  Given type: {}'.format(
        type(units)))

  def dnn_logit_fn(features, mode):
    with variable_scope.variable_scope(
        'input_from_feature_columns',
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner):
      print("---------------------")
      print(features["app_download_count"])
      print(feature_columns)
      net = feature_column_lib.input_layer(
          features=features, feature_columns=feature_columns)
      queryNet = feature_column_lib.input_layer(
        features=features, feature_columns=[feature_columns[_COLS_IDX_QUERY]])
      appNameNet = feature_column_lib.input_layer(
        features=features, feature_columns=[feature_columns[_COLS_IDX_APPNAME]])
      noDotQueryApp = tf.multiply(queryNet, appNameNet)
      dotQueryApp = tf.reduce_sum(noDotQueryApp, 1, keep_dims=True)  # dot product cross
      net = tf.concat([net, dotQueryApp], 1)
    for layer_id, num_hidden_units in enumerate(hidden_units):
      with variable_scope.variable_scope(
          'hiddenlayer_%d' % layer_id, values=(net,)) as hidden_layer_scope:
        net = core_layers.dense(
            net,
            units=num_hidden_units,
            activation=activation_fn,
            kernel_initializer=init_ops.glorot_uniform_initializer(),
            # kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=0.1, scope=None),
            name=hidden_layer_scope)
        if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
          net = core_layers.dropout(net, rate=dropout, training=True)
      _add_hidden_layer_summary(net, hidden_layer_scope.name)

    if isinstance(units, int):
      with variable_scope.variable_scope(
          'logits', values=(net,)) as logits_scope:
        logits = core_layers.dense(
            net,
            units=units,
            activation=None,
            kernel_initializer=init_ops.glorot_uniform_initializer(),
            name=logits_scope)
      _add_hidden_layer_summary(logits, logits_scope.name)
    else:
      logits = []
      for head_index, logits_dimension in enumerate(units):
        with variable_scope.variable_scope(
            'logits_head_{}'.format(head_index), values=(net,)) as logits_scope:
          these_logits = core_layers.dense(
              net,
              units=logits_dimension,
              activation=None,
              kernel_initializer=init_ops.glorot_uniform_initializer(),
              name=logits_scope)
        _add_hidden_layer_summary(these_logits, logits_scope.name)
        logits.append(these_logits)
    return logits
  return dnn_logit_fn


_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


def _check_no_sync_replicas_optimizer(optimizer):
  if isinstance(optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
    raise ValueError(
        'SyncReplicasOptimizer does not support multi optimizers case. '
        'Therefore, it is not supported in DNNLinearCombined model. '
        'If you want to use this optimizer, please use either DNN or Linear '
        'model.')


def _linear_learning_rate(num_linear_feature_columns):
  default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
  return min(_LINEAR_LEARNING_RATE, default_learning_rate)


def _add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)


def _dnn_linear_combined_model_fn(
    features, labels, mode, head,
    linear_feature_columns=None, linear_optimizer='Ftrl',
    dnn_feature_columns=None, dnn_optimizer='Adagrad', dnn_hidden_units=None,
    dnn_activation_fn=nn.relu, dnn_dropout=None,
    input_layer_partitioner=None, config=None):
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))
  if not linear_feature_columns and not dnn_feature_columns:
    raise ValueError(
        'Either linear_feature_columns or dnn_feature_columns must be defined.')
  num_ps_replicas = config.num_ps_replicas if config else 0
  input_layer_partitioner = input_layer_partitioner or (
      partitioned_variables.min_max_variable_partitioner(
          max_partitions=num_ps_replicas,
          min_slice_size=64 << 20))

  # Build DNN Logits.
  dnn_parent_scope = 'dnn'

  if not dnn_feature_columns:
    dnn_logits = None
  else:
    dnn_optimizer = optimizers.get_optimizer_instance(
        dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
    _check_no_sync_replicas_optimizer(dnn_optimizer)
    if not dnn_hidden_units:
      raise ValueError(
          'dnn_hidden_units must be defined when dnn_feature_columns is '
          'specified.')
    dnn_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas))
    with variable_scope.variable_scope(
        dnn_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=dnn_partitioner):

      dnn_logit_fn = _dnn_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          hidden_units=dnn_hidden_units,
          feature_columns=dnn_feature_columns,
          activation_fn=dnn_activation_fn,
          dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner)
      dnn_logits = dnn_logit_fn(features=features, mode=mode)

  linear_parent_scope = 'linear'

  if not linear_feature_columns:
    linear_logits = None
  else:
    linear_optimizer = optimizers.get_optimizer_instance(
        linear_optimizer,
        learning_rate=_linear_learning_rate(len(linear_feature_columns)))
    _check_no_sync_replicas_optimizer(linear_optimizer)
    with variable_scope.variable_scope(
        linear_parent_scope,
        values=tuple(six.itervalues(features)),
        partitioner=input_layer_partitioner) as scope:
      logit_fn = linear._linear_logit_fn_builder(  # pylint: disable=protected-access
          units=head.logits_dimension,
          feature_columns=linear_feature_columns)
      linear_logits = logit_fn(features=features)
      _add_layer_summary(linear_logits, scope.name)

  # Combine logits and build full model.
  if dnn_logits is not None and linear_logits is not None:
    logits = dnn_logits + linear_logits
  elif dnn_logits is not None:
    logits = dnn_logits
  else:
    logits = linear_logits

  def _train_op_fn(loss):
    train_ops = []
    global_step = training_util.get_global_step()
    if dnn_logits is not None:
      train_ops.append(
          dnn_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=dnn_parent_scope)))
    if linear_logits is not None:
      train_ops.append(
          linear_optimizer.minimize(
              loss,
              var_list=ops.get_collection(
                  ops.GraphKeys.TRAINABLE_VARIABLES,
                  scope=linear_parent_scope)))

    train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
      with ops.colocate_with(global_step):
        return state_ops.assign_add(global_step, 1)

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      train_op_fn=_train_op_fn,
      logits=logits)

class DNNLinearCombinedClassifier(estimator.Estimator):
  def __init__(self,
               model_dir=None,
               linear_feature_columns=None,
               linear_optimizer='Ftrl',
               dnn_feature_columns=None,
               dnn_optimizer='Adagrad',
               dnn_hidden_units=None,
               dnn_activation_fn=nn.relu,
               dnn_dropout=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               input_layer_partitioner=None,
               config=None):
    linear_feature_columns = linear_feature_columns or []
    dnn_feature_columns = dnn_feature_columns or []
    self._feature_columns = (
        list(linear_feature_columns) + list(dnn_feature_columns))
    if not self._feature_columns:
      raise ValueError('Either linear_feature_columns or dnn_feature_columns '
                       'must be defined.')
    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary)
    def _model_fn(features, labels, mode, config):
      return _dnn_linear_combined_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          linear_feature_columns=linear_feature_columns,
          linear_optimizer=linear_optimizer,
          dnn_feature_columns=dnn_feature_columns,
          dnn_optimizer=dnn_optimizer,
          dnn_hidden_units=dnn_hidden_units,
          dnn_activation_fn=dnn_activation_fn,
          dnn_dropout=dnn_dropout,
          input_layer_partitioner=input_layer_partitioner,
          config=config)

    super(DNNLinearCombinedClassifier, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


class WideAndDeepModel(object):
    def __init__(self,
            model_type = "wide",
            model_dir = None,
            feature_size = 10000000,
            run_config = None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        if run_config is None:
            self.run_config = self.build_run_config()
        else:
            self.run_config = run_config

    def build_run_config(self):
        run_config = tf.contrib.learn.RunConfig(
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 100,               # Save summaries every this many steps
                save_checkpoints_secs = 600,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 5,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 100)             # The frequency, in number of global steps
        return run_config
    #这里改
    def build_feature_dict(self):
      feature_dict = {}
      feature_dict["query_indices"] = tf.placeholder(dtype=tf.string, shape=(None, None))
      feature_dict["appid"] = tf.placeholder(dtype=tf.int32, shape=(None, 1))
      feature_dict["appname_indices"] = tf.placeholder(dtype=tf.string, shape=(None, None))
      feature_dict["appfeature"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["week"] = tf.placeholder(dtype=tf.string, shape=(None, None))
      feature_dict["lsi"] = tf.placeholder(dtype=tf.float32, shape=(None, None))
      feature_dict["app_download_count"] = tf.placeholder(dtype=tf.float32, shape=(None, None))
      feature_dict["imei"] = tf.placeholder(dtype=tf.string, shape=(None, 1))
      feature_dict["install_tags_indices"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["uninstall_tags_indices"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["phonetype"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["search_indices"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["appstore_search_indices"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["citys"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      feature_dict["appuse_length"] = tf.placeholder(dtype=tf.int32, shape=(None, None))
      return feature_dict

    def build_feature_columns(self):

      query_features = tf.feature_column.categorical_column_with_hash_bucket(
        key="query_indices",
        hash_bucket_size=5000000)
      query_embedding_features = tf.feature_column.embedding_column(
        categorical_column=query_features,
        dimension=128,
        combiner="sqrtn")

      appid_features = tf.feature_column.categorical_column_with_hash_bucket(
        key="appid",
        hash_bucket_size=200000,
        dtype=tf.int64)
      appid_embedding_features = tf.feature_column.embedding_column(
        categorical_column=appid_features,
        dimension=128,
        combiner="sqrtn")

      appname_features = tf.feature_column.categorical_column_with_hash_bucket(
        key="appname_indices",
        hash_bucket_size=200000)
      appname_embedding_features = tf.feature_column.embedding_column(
        categorical_column=appname_features,
        dimension=128,
        combiner="sqrtn")

      app_features = tf.feature_column.categorical_column_with_identity(
        key="appfeature",
        num_buckets=2000,
        default_value=0)
      app_embedding_features = tf.feature_column.embedding_column(
        categorical_column=app_features,
        dimension=32,
        combiner="sqrtn")

	  #cross feature for week and hour
      week_and_hour_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009',
                                    '010', '011', '012', '013', '014', '015', '016', '017', '018', '019',
                                    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                                    '110', '111', '112', '113', '114', '115', '116', '117', '118', '119',
                                    '200', '201', '202', '203', '204', '205', '206', '207', '208', '209',
                                    '210', '211', '212', '213', '214', '215', '216', '217', '218', '219',
                                    '300', '301', '302', '303', '304', '305', '306', '307', '308', '309',
                                    '310', '311', '312', '313', '314', '315', '316', '317', '318', '319',
                                    '400', '401', '402', '403', '404', '405', '406', '407', '408', '409',
                                    '410', '411', '412', '413', '414', '415', '416', '417', '418', '419',
                                    '500', '501', '502', '503', '504', '505', '506', '507', '508', '509',
                                    '510', '511', '512', '513', '514', '515', '516', '517', '518', '519',
                                    '600', '601', '602', '603', '604', '605', '606', '607', '608', '609',
                                    '610', '611', '612', '613', '614', '615', '616', '617', '618', '619',
                                    '020', '021', '022', '023', '120', '121', '122', '123', '220', '221', '222', '223',
                                    '320', '321', '322', '323', '420', '421', '422', '423', '520', '521', '522', '523',
                                    '620', '621', '622', '623']
      week_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="week",
        vocabulary_list = week_and_hour_list,
        default_value=0)
      week = tf.feature_column.indicator_column(week_column)
      
      lsi = tf.feature_column.numeric_column(key="lsi", shape=(5,))
      
	  #the number of user who downloaded a app
      app_download_count = tf.feature_column.numeric_column(
        key="app_download_count",
        shape=(6,))
		

      install_tags_indices = tf.feature_column.categorical_column_with_identity(
        key="install_tags_indices",
        num_buckets=2000,
        default_value=0)
      install_tags_embedding = tf.feature_column.embedding_column(
        categorical_column=install_tags_indices,
        dimension=32,
        combiner="sqrtn")

      uninstall_tags_indices = tf.feature_column.categorical_column_with_identity(
        key="uninstall_tags_indices",
        num_buckets=2700,
        default_value=0)
      uninstall_tags_embedding = tf.feature_column.embedding_column(
        categorical_column=uninstall_tags_indices,
        dimension=32,
        combiner="sqrtn")

      phonetype = tf.feature_column.categorical_column_with_identity(
        key="phonetype",
        num_buckets=30,
        default_value=0)
      phonetype_embedding = tf.feature_column.embedding_column(
        categorical_column=phonetype,
        dimension=4,
        combiner="sqrtn")

      citys = tf.feature_column.categorical_column_with_identity(
        key="citys",
        num_buckets=30,
        default_value=0)
      citys_embedding = tf.feature_column.embedding_column(
        categorical_column=citys,
        dimension=4,
        combiner="sqrtn")

      search_indices = tf.feature_column.categorical_column_with_identity(
        key="search_indices",
        num_buckets=320,
        default_value=0)
      search_embedding = tf.feature_column.embedding_column(
        categorical_column=search_indices,
        dimension=32,
        combiner="sqrtn")

      appstore_search_indices = tf.feature_column.categorical_column_with_identity(
        key="appstore_search_indices",
        num_buckets=1600,
        default_value=0)
      appstore_search_embedding = tf.feature_column.embedding_column(
        categorical_column=appstore_search_indices,
        dimension=32,
        combiner="sqrtn")

      appuse_length = tf.feature_column.categorical_column_with_identity(
        key="appuse_length",
        num_buckets=1020,
        default_value=0)
      appuse_length_embedding = tf.feature_column.embedding_column(
        categorical_column=appuse_length,
        dimension=32,
        combiner="sqrtn")

      linear_feature_columns = []
      
      dnn_feature_columns = [query_embedding_features, appname_embedding_features, appid_embedding_features,
                             app_embedding_features,week,lsi,app_download_count,install_tags_embedding,
                             uninstall_tags_embedding, phonetype_embedding, search_embedding,
                             appstore_search_embedding, citys_embedding, appuse_length_embedding]
      

      return (linear_feature_columns, dnn_feature_columns)

    def build_linear_optimizer(self):
        linear_optimizer = tf.train.FtrlOptimizer(
                learning_rate = 0.1,
                learning_rate_power = -0.5,
                initial_accumulator_value = 0.1,
                l1_regularization_strength = 0.4,
                l2_regularization_strength = 0.4)
        return linear_optimizer

    def build_dnn_optimizer(self):
        dnn_optimizer = tf.train.AdagradOptimizer(
                learning_rate = 0.01,
                initial_accumulator_value = 0.1)
        return dnn_optimizer

    def build_estimator(self):
        linear_optimizer = self.build_linear_optimizer()
        dnn_optimizer = self.build_dnn_optimizer()
        dnn_hidden_units = [256, 128, 32]
        (linear_feature_columns, dnn_feature_columns) = self.build_feature_columns()

        if self.model_type == "wide":
            model = tf.estimator.LinearClassifier(
                    feature_columns = linear_feature_columns,
                    model_dir = self.model_dir,
                    optimizer = linear_optimizer,
                    config = self.run_config)
        elif self.model_type == "deep":
            model = tf.estimator.DNNClassifier(
                    hidden_units=dnn_hidden_units,
                    feature_columns=dnn_feature_columns,
                    model_dir=self.model_dir,
                    optimizer=dnn_optimizer,
                    activation_fn=tf.nn.relu,
                    dropout=None,
                    config=self.run_config)
        elif self.model_type == "wide_and_deep":
            model = DNNLinearCombinedClassifier(
                    model_dir = self.model_dir,
                    linear_feature_columns = linear_feature_columns,
                    linear_optimizer = linear_optimizer,
                    dnn_feature_columns = dnn_feature_columns,
                    dnn_optimizer = dnn_optimizer,
                    dnn_hidden_units = dnn_hidden_units,
                    dnn_activation_fn = tf.nn.relu,
                    dnn_dropout=None,
                    config = self.run_config)
        else:
            logging.error("Unsupported model type: %s" % (self.model_type))
        return model


####################################################################################################


class WideAndDeepInputPipeline(object):
    def __init__(self, input_files, batch_size = 1000, num_epochs = 1, shuffle = True):
        self.batch_size = batch_size
        self.input_files = input_files

        input_file_list = []
        for input_file in self.input_files:
            if len(input_file) > 0:
                input_file_list.append(tf.train.match_filenames_once(input_file))
        self.filename_queue = tf.train.string_input_producer(
                tf.concat(input_file_list, axis = 0),
                num_epochs = None,     # strings are repeated num_epochs
                shuffle = shuffle,     # strings are randomly shuffled within each epoch
                capacity = 512)

        self.reader_csv = tf.TextLineReader(skip_header_lines = 0)
        self.reader_tfrecord = tf.TFRecordReader()

        (self.column_dict, self.column_defaults) = self.build_column_format()

    """
    构建特征表的列名
    """
    def build_column_format(self):
      column_dict = {"label": 0, "query_indices": 1, "appid": 2, "appname_indices": 3, "appfeature": 4,
                     "week": 5, "lsi": 6, "app_download_count": 7, "imei": 8, "install_tags_indices": 9, "uninstall_tags_indices": 10,
                     "phonetype": 11, "search_indices": 12, "appstore_search_indices": 13,
                     "citys": 14, "appuse_length": 15}
      column_defaults = [['']] * len(column_dict)
      column_defaults[column_dict["label"]] = [0.0]
      column_defaults[column_dict["query_indices"]] = ['0']
      column_defaults[column_dict["appid"]] = ['0']
      column_defaults[column_dict["appname_indices"]] = ['0']
      column_defaults[column_dict["appfeature"]] = ['0']
      column_defaults[column_dict["week"]] = ['0']
      column_defaults[column_dict["lsi"]] = ['0']
      column_defaults[column_dict["app_download_count"]] = ['0']
      column_defaults[column_dict["imei"]] = ['0']
      column_defaults[column_dict["install_tags_indices"]] = ['0']
      column_defaults[column_dict["uninstall_tags_indices"]] = ['0']
      column_defaults[column_dict["phonetype"]] = ['0']
      column_defaults[column_dict["search_indices"]] = ['0']
      column_defaults[column_dict["appstore_search_indices"]] = ['0']
      column_defaults[column_dict["citys"]] = ['0']
      column_defaults[column_dict["appuse_length"]] = ['0']

      return (column_dict, column_defaults)

#chang the spase_tenso to dense
    def string_to_number(self, string_tensor, dtype = tf.int32):
        number_values = tf.string_to_number(
                string_tensor = string_tensor.values,
                out_type = dtype)
        number_tensor = tf.SparseTensor(
                indices = string_tensor.indices,
                values = number_values,
                dense_shape = string_tensor.dense_shape)
        return number_tensor

    def string_to_number_dense(self, string_tensor, dtype = tf.int32):
        number_values = tf.string_to_number(
                string_tensor = string_tensor.values,
                out_type = dtype)
        number_tensor = tf.SparseTensor(
                indices = string_tensor.indices,
                values = number_values,
                dense_shape = string_tensor.dense_shape)
        number_dense_tensor = tf.sparse_tensor_to_dense(number_tensor)
        return number_dense_tensor


    def get_next_batch(self):
      if FLAGS.file_format == "tfrecord":
        (_, records) = self.reader_tfrecord.read_up_to(queue=self.filename_queue, num_records=self.batch_size)
        samples = tf.parse_example(
          records,
          features=get_features()
        )

        label = tf.unstack(samples['label'], axis=1)[0]  # 这里注意是否要转
        feature_dict = {}
        for (key, value) in self.column_dict.items():
          if key != 'label':
            feature_dict[key] = samples[key]

      elif FLAGS.file_format == "csv":
        (_, records) = self.reader_csv.read_up_to(queue=self.filename_queue, num_records=self.batch_size)
        samples = tf.decode_csv(records, record_defaults=self.column_defaults, field_delim=',')
        label = tf.cast(samples[self.column_dict["label"]], dtype=tf.int32)
        feature_dict = {}

        for (key, value) in self.column_dict.items():
          if key == "label" or value < 0 or value >= len(samples):
            continue
          if key in ["query_indices", "appname_indices", "week", "imei"]:
            feature_dict[key] = tf.string_split(samples[value], delimiter=';')
          if key in ["appid", "appfeature", "install_tags_indices", "uninstall_tags_indices",
                     "phonetype", "search_indices", "appstore_search_indices", "citys", "appuse_length"]:
            feature_dict[key] = self.string_to_number(
              tf.string_split(samples[value], delimiter=';'),
              dtype=tf.int32)
          if key in ["lsi", "app_download_count"]:
            print("########################")
            print("key is {k} seconds".format(k=key))
            print(tf.string_split(samples[value], delimiter=';'))
            print("########################")
            feature_dict[key] = self.string_to_number_dense(
              tf.string_split(samples[value], delimiter=';'),
              dtype=tf.float32)
        print(feature_dict["app_download_count"])
        #feature_dict["app_download_count"] = tf.Print(feature_dict["app_download_count"],[feature_dict["app_download_count"],feature_dict["app_download_count"].shape,'shape of app_download_count'],message='Debug message:',summarize=100)  
      else:
        raise ValueError("file_format must be csv or tfrecord")

      return feature_dict, label


####################################################################################################

def get_features():
  features = {
    'label': tf.FixedLenFeature([1], dtype=tf.int64, default_value=0),
    'query_indices': tf.VarLenFeature(tf.string),
    'appid': tf.VarLenFeature(dtype=tf.int64),
    'appname_indices': tf.VarLenFeature(tf.string),
    'appfeature': tf.VarLenFeature(tf.int64),
    #'week': tf.VarLenFeature(dtype=tf.int64),
    'week': tf.VarLenFeature(dtype=tf.string),
    'lsi': tf.FixedLenFeature([5], dtype=tf.float32),
    'imei': tf.FixedLenFeature([1], dtype=tf.string, default_value='0'),
    'app_download_count': tf.FixedLenFeature([6], dtype=tf.float32),
    'install_tags_indices': tf.VarLenFeature(tf.int64),
    'uninstall_tags_indices': tf.VarLenFeature(tf.int64),
    'phonetype': tf.VarLenFeature(tf.int64),
    'search_indices': tf.VarLenFeature(tf.int64),
    'appstore_search_indices': tf.VarLenFeature(tf.int64),
    'citys': tf.VarLenFeature(tf.int64),
    'appuse_length': tf.VarLenFeature(tf.int64)
  }
  return features


def train_input_fn():
    train_input_files = FLAGS.train_data.strip().split(',')
    random.shuffle(train_input_files)
    train_input_pipeline = WideAndDeepInputPipeline(
            train_input_files,
            batch_size = FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs,
            shuffle=True)
    return train_input_pipeline.get_next_batch()


def train_model():
    if FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        #tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    estimator.train(
            input_fn = lambda: train_input_fn(),
            steps = FLAGS.train_steps)


def eval_input_fn():
    eval_input_files = FLAGS.eval_data.strip().split(',')
    eval_input_pipeline = WideAndDeepInputPipeline(
            eval_input_files,
            batch_size=131072,
            num_epochs=1,
            shuffle=False)
    return eval_input_pipeline.get_next_batch()


def eval_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    # Get the checkpoint path for evaluation
    checkpoint_path = None # The latest checkpoint in model dir
    if FLAGS.eval_ckpt_id > 0:
        state = tf.train.get_checkpoint_state(
                checkpoint_dir = FLAGS.model_dir,
                latest_filename = "checkpoint")
        if state and state.all_model_checkpoint_paths:
            if FLAGS.eval_ckpt_id < len(state.all_model_checkpoint_paths):
                pos = -(1 + FLAGS.eval_ckpt_id)
                checkpoint_path = state.all_model_checkpoint_paths[pos]
            else:
                logging.warn("not find checkpoint id %d in %s" % (FLAGS.eval_ckpt_id, FLAGS.model_dir))
                checkpoint_path = None
    logging.info("checkpoint path: %s" % (checkpoint_path))
    eval_name = '' if checkpoint_path is None else str(FLAGS.eval_ckpt_id)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    eval_result = estimator.evaluate(
            input_fn = lambda: eval_input_fn(),
            steps = FLAGS.eval_steps,
            checkpoint_path = checkpoint_path,
            name = eval_name)
    print(eval_result)

def export_savedmodel():
  if not tf.gfile.Exists(FLAGS.model_dir):
    logging.error("not find model dir: %s" % (FLAGS.model_dir))
    sys.exit(1)

  model = WideAndDeepModel(
    model_type=FLAGS.model_type,
    model_dir=FLAGS.model_dir,
    feature_size=FLAGS.feature_size)
  estimator = model.build_estimator()

  if FLAGS.savedmodel_mode == "raw":
    features = model.build_feature_dict()
    export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
      features=features,
      default_batch_size=None)
  elif FLAGS.savedmodel_mode == "parsing":
    (linear_feature_columns, dnn_feature_columns) = model.build_feature_columns()
    feature_columns = linear_feature_columns + dnn_feature_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
      feature_spec=feature_spec, #这种是由解析生成
      default_batch_size=None)
  else:
    logging.error("unsupported savedmodel mode: %s" % (FLAGS.savedmodel_mode))
    sys.exit(1)

  export_dir = estimator.export_savedmodel(
    export_dir_base=FLAGS.export_savedmodel,
    serving_input_receiver_fn=lambda: export_input_fn(),
    assets_extra=None,
    as_text=False,
    checkpoint_path=None)

def predict_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size)
    estimator = model.build_estimator()
    predict = estimator.predict(
            input_fn = lambda: eval_input_fn(),
            predict_keys = None,
            hooks = None,
            checkpoint_path = None)
    results = list(predict)
    sum_score = 0.0
    for i in range(0, len(results)):
        result = results[i]
        sum_score = sum_score + result["logistic"][0]
        print("count: %d, score: %f" % (i + 1, result["logistic"][0]))
    print("total count: %d, average score: %f" % (len(results), sum_score / len(results)))


####################################################################################################

def label_to_dense(label,dim=2):
  #dim is the 类别数量
  batch_size = tf.size(label)
  label = tf.expand_dims(label, 1)
  indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
  concated = tf.concat([indices, label], 1)
  label = tf.sparse_to_dense(concated, tf.stack([batch_size, dim]), 1.0, 0.0)
  return label

def test_input_pipeline():
    logging.info("train data: %s" % (FLAGS.train_data))
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
            train_input_files,
            batch_size = 1)
    feature, label = train_input_pipeline.get_next_batch()
    #label = label_to_dense(label=label, dim=2)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        features, labels = sess.run([feature, label])

        #print(labels)
        print(features["imei"])
        coord.request_stop()
        coord.join(threads)


####################################################################################################


class WideAndDeepDistributedRunner(object):
    def __init__(self,
            model_type = "wide",
            model_dir = None,
            feature_size = 10000000,
            schedule = "train",
            worker_hosts = None,
            ps_hosts = None,
            task_type = None,
            task_index = None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        self.schedule = schedule
        self.worker_hosts = worker_hosts.strip().split(",")
        self.ps_hosts = ps_hosts.strip().split(",")
        self.task_type = task_type
        self.task_index = task_index

        self.run_config = self.build_run_config()
        self.hparams = self.build_hparams()


    def build_run_config(self):
        cluster = {"worker": self.worker_hosts, "ps": self.ps_hosts}
        task = {"type": self.task_type, "index": self.task_index}
        environment = {"environment": "cloud"}
        os.environ["TF_CONFIG"] = json.dumps({"cluster": cluster, "task": task, "environment": environment})

        run_config = tf.contrib.learn.RunConfig(
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 1000,              # Save summaries every this many steps
                save_checkpoints_secs = 600,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 5,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 1000,            # The frequency, in number of global steps
                model_dir = self.model_dir)             # Directory where model parameters, graph etc are saved
        return run_config


    def build_hparams(self):
        hparams = tf.contrib.training.HParams(
                eval_metrics = None,
                train_steps = FLAGS.train_steps,
                eval_steps = FLAGS.eval_steps,
                eval_delay_secs = 5,
                min_eval_frequency = 50000)
        return hparams


    def build_experiment(self, run_config, hparams):
        model = WideAndDeepModel(
                model_type = self.model_type,
                model_dir = self.model_dir,
                feature_size = self.feature_size,
                run_config = run_config)
        return tf.contrib.learn.Experiment(
                estimator = model.build_estimator(),
                train_input_fn = lambda: train_input_fn(),
                eval_input_fn = lambda: eval_input_fn(),
                eval_metrics = hparams.eval_metrics,
                train_steps = hparams.train_steps,
                eval_steps = hparams.eval_steps,
                eval_delay_secs = hparams.eval_delay_secs,
                min_eval_frequency = hparams.min_eval_frequency)


    def run(self):
        tf.contrib.learn.learn_runner.run(
                experiment_fn = self.build_experiment,
                output_dir = None, # Deprecated, must be None
                schedule = self.schedule,
                run_config = self.run_config,
                hparams = self.hparams)


####################################################################################################

def local_run():
    if FLAGS.job_type == "train":
        train_model()
    elif FLAGS.job_type == "eval":
        eval_model()
    elif FLAGS.job_type == "train_and_eval":
        train_model()
        eval_model()
    elif FLAGS.job_type == "export_savedmodel":
        export_savedmodel()
    elif FLAGS.job_type == "predict":
        predict_model()
    else:
        logging.error("unsupported job type: %s" % (FLAGS.job_type))
        sys.exit(1)

def distributed_run():
    if FLAGS.task_type == "worker" and FLAGS.task_index == 0 \
            and (FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval") \
            and FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        #tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    schedule = None
    schedule_dict = {"train": "train", "eval": "evaluate", "train_and_eval": "train_and_evaluate"}
    if FLAGS.task_type == "ps":
        schedule = "run_std_server"
    elif FLAGS.task_type == "worker":
        schedule = schedule_dict.get(FLAGS.job_type, None)
        if FLAGS.job_type == "train_and_eval" and FLAGS.task_index != 0:
            schedule = "train"  # only the first worker runs evaluation
    else:
        logging.error("unsupported task type: %s" % (FLAGS.task_type))
        sys.exit(1)
    logging.info("schedule: %s" % (schedule))

    runner = WideAndDeepDistributedRunner(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            schedule = schedule,
            worker_hosts = FLAGS.worker_hosts,
            ps_hosts = FLAGS.ps_hosts,
            task_type = FLAGS.task_type,
            task_index = FLAGS.task_index)
    runner.run()


def main():
    # print commandline arguments
    logging.info("run mode: %s" % (FLAGS.run_mode))
    if FLAGS.run_mode == "distributed":
        logging.info("worker hosts: %s" % (FLAGS.worker_hosts))
        logging.info("ps hosts: %s" % (FLAGS.ps_hosts))
        logging.info("task type: %s, task index: %d" % (FLAGS.task_type, FLAGS.task_index))
    logging.info("job type: %s" % (FLAGS.job_type))
    if FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval":
        logging.info("train data: %s" % (FLAGS.train_data))
        logging.info("cold start: %s" % (FLAGS.cold_start))
    if FLAGS.job_type in ["eval", "train_and_eval", "predict"]:
        logging.info("eval data: %s" % (FLAGS.eval_data))
        logging.info("eval ckpt id: %s" % (FLAGS.eval_ckpt_id))
    logging.info("model dir: %s" % (FLAGS.model_dir))
    logging.info("model type: %s" % (FLAGS.model_type))
    logging.info("feature size: %s" % (FLAGS.feature_size))
    logging.info("batch size: %s" % (FLAGS.batch_size))

    if FLAGS.run_mode == "local":
        local_run()
    elif FLAGS.run_mode == "distributed":
        if FLAGS.job_type == "export_savedmodel" or FLAGS.job_type == "predict":
            logging.error("job type export_savedmodel and predict does not support distributed run mode")
            sys.exit(1)
        if FLAGS.job_type in ["eval", "train_and_eval"] and FLAGS.eval_ckpt_id != 0:
            logging.error("eval_ckpt_id does not support distributed run mode")
            sys.exit(1)
        distributed_run()
    else:
        logging.error("unsupported run mode: %s" % (FLAGS.run_mode))
        sys.exit(1)
    prefix = "" if FLAGS.run_mode == "local" else "%s:%d " % (FLAGS.task_type, FLAGS.task_index)
    logging.info("%scompleted" % (prefix))


def test():
    test_input_pipeline()


if __name__ == "__main__":
    main()
    #test()
