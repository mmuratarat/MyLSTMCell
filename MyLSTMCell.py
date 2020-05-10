import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMCell
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class MyLSTMCell(tf.contrib.rnn.LSTMCell):
    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None, name=None, var_given=False, kernel=None, bias=None):

        super(MyLSTMCell, self).__init__(num_units,
                 use_peepholes=use_peepholes, cell_clip=cell_clip,
                 initializer=initializer, num_proj=num_proj, proj_clip=proj_clip,
                 num_unit_shards=num_unit_shards, num_proj_shards=num_proj_shards,
                 forget_bias=forget_bias, state_is_tuple=state_is_tuple,
                 activation=activation, reuse=reuse, name=name)

        self.var_given = var_given
        if self.var_given:
            self._kernel = kernel
            self._bias = bias


    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)
        if self.var_given:
            # self._kernel and self._bias are already added in init
            pass
        else:
            self._kernel = self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + h_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[4 * self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True