import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers

mapping = {}
def register(name):
    # print("register:",name)
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


@register("encode_tau_state")
def encode_tau_state(num_hidden=64,**kwargs):#
    """Used to register custom network type used by Baselines for Overcooked"""

    if "network_kwargs" in kwargs.keys():
        params = kwargs["network_kwargs"]
    else:
        params = kwargs

    num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
    size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
    num_filters = params["NUM_FILTERS"]
    num_convs = params["NUM_CONV_LAYERS"]

    def network_fn(X, latent=None):
        print("encode_tau_state.shape", X.shape)
        batch_size=X.shape[0]
        last_dim=X.shape[-1]

        conv_out = tf.layers.conv2d(
            inputs=X,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )

        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )

        out = tf.layers.flatten(conv_out)

        print("out.shape:", out.shape)  # (2000,150)
        if latent != None:
            latent_last_dim=latent.shape[-1]
            latent=tf.reshape(latent,(-1,latent_last_dim))
            out = tf.concat([out, latent], axis=1)
            for _ in range(num_hidden_layers-1):
                out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)
            out=tf.layers.dense(out,num_hidden, activation=tf.nn.leaky_relu)





        # NOTE: not sure if not supposed to add linear layer. I think it is though,
        # as things work and similar to code in baseline/models.py? Maybe double check later.

        # To check how many parameters uncomment next line
        # num_tf_params()
        return out

    return network_fn
def conv_network(num_filters,num_convs,num_hidden_layers,size_hidden_layers):
    def network_fn(X):
        conv_out = tf.layers.conv2d(
            inputs=X,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )

        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )
        print("out.shape: in conv_network conv embd:",conv_out.shape)

        out = tf.layers.flatten(conv_out)
        # for _ in range(num_hidden_layers):
        #     out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)
        return out
    return network_fn


def self_attn(x,length):

        # self.Q = th.nn.Linear(128, 128)
        # self.K = th.nn.Linear(128, 128)
        # self.V = th.nn.Linear(128, 128)

        q=tf.layers.dense(x,64, activation=tf.nn.sigmoid,name='Q')
        k = tf.layers.dense(x, 64, activation=tf.nn.sigmoid, name='K')
        v = tf.layers.dense(x, 64, activation=tf.nn.sigmoid, name='V')
        q=tf.reshape(q,(-1,length,64))
        k=tf.reshape(k,(-1,length,64))
        weights=tf.nn.softmax(tf.matmul(q,tf.transpose(k,(0,2,1)))/np.sqrt(64),axis=-1)#(B,len,len)
        out=tf.sigmoid(tf.matmul(weights,v))

        # v=tf.reshape(v,(-1,length,64))
        #
        # q = self.Q(x).view(-1, 10, 128)
        # k = self.K(x).view(-1, 10, 128)
        # k = th.cat([th.t(key).view(-1, 128, 10) for key in k], dim=0)
        # v = self.V(x).view(-1, 10, 128)
        # weights = th.softmax(th.bmm(q, k) / np.sqrt(128), dim=1)
        # x = th.bmm(weights, v).reshape(-1, 10, 128)
        return out


@register("conv_and_mlp_embd")
def conv_network_fn_embd(**kwargs):#
    """Used to register custom network type used by Baselines for Overcooked"""

    if "network_kwargs" in kwargs.keys():
        params = kwargs["network_kwargs"]
    else:
        params = kwargs

    num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
    size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
    num_filters = params["NUM_FILTERS"]
    num_convs = params["NUM_CONV_LAYERS"]
    latent_dim=params["LATENT_DIM"]
    length=params['LENGTH']
    train_mode=params['TRAIN_MODE']

    def network_fn(X, latent=None):

        print("conv_and_mlp.shape", X.shape)
        batch_size=X.shape[0]
        last_dim=X.shape[-1]

        xshape=X.shape
        print(xshape[2:])
        if length!=1:
            X=tf.reshape(X,shape=(xshape[0]*xshape[1],xshape[2],xshape[3],xshape[4]))

        conv_out = tf.layers.conv2d(
            inputs=X,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )
        #(B,5,5,25)
        for i in range(0, num_convs - 1):
            activate=tf.nn.leaky_relu
            if i ==num_convs-2:
                activate=tf.identity
            padding = "same" if i < num_convs - 2 else "valid"
            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=activate,
                name="conv_{}".format(i)
            )
        print("out.shape: in conv_network conv embd:", conv_out.shape)

        out = tf.layers.flatten(conv_out)
        # one-hot action->v
        print("out.shape:", out.shape)  # (2000,150) 64

        if latent != None:
            latent=tf.reshape(latent,(batch_size*length,-1))
            latent_last_dim=latent.shape[-1]

            if latent_last_dim<=2:
                one_hot_labels = tf.one_hot(latent, 6)#20000,1,6
                print("one_hot_labels.shape:",one_hot_labels.shape)
                latent = tf.layers.dense(one_hot_labels, 6, activation=tf.identity, name='word_embd')
                latent = tf.reshape(latent, (-1, latent_last_dim*6))
            out = tf.concat([out, latent], axis=1) #without action
            out=tf.nn.leaky_relu(out)

            with tf.variable_scope('dense_in_context_encoder', reuse=tf.AUTO_REUSE):

                for _ in range(num_hidden_layers - 1):
                    out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)
                out = tf.layers.dense(out, latent_dim, activation=tf.nn.leaky_relu)

        if length>1:
            if 'MLP' in train_mode:
                out = tf.reshape(out, (batch_size, -1))
                for _ in range(num_hidden_layers):
                    out = tf.layers.dense(out, latent_dim, activation=tf.nn.leaky_relu)
            if 'SA' in train_mode:
                with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
                    out = tf.reshape(out, (-1, length, latent_dim))
                    out = self_attn(out, length)
                    out = tf.reduce_mean(out, axis=1)





        # NOTE: not sure if not supposed to add linear layer. I think it is though,
        # as things work and similar to code in baseline/models.py? Maybe double check later.

        # To check how many parameters uncomment next line
        # num_tf_params()
        return out

    return network_fn

@register("conv_and_mlp_new")
def conv_network_fn_new(**kwargs):#
    """Used to register custom network type used by Baselines for Overcooked"""

    if "network_kwargs" in kwargs.keys():
        params = kwargs["network_kwargs"]
    else:
        params = kwargs
    if 'TRAIN_MODE' in params.keys():
        train_mode=params['TRAIN_MODE']

    num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
    size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
    num_filters = params["NUM_FILTERS"]
    num_convs = params["NUM_CONV_LAYERS"]
    # BN=params['BN']
    length=params['LENGTH']
    latent_dim=params["LATENT_DIM"]

    def network_fn(X, latent=None):
        print("conv_and_mlp.shape", X.shape)
        xshape=X.shape
        batch_size=xshape[0]

        conv_out = tf.layers.conv2d(
            inputs=X,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )

        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"

            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )

        out = tf.layers.flatten(conv_out)
        print("out.shape:", out.shape)  # (2000,150)


        print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        if latent != None:
            out = tf.concat([out, latent], axis=1)#[a1,a2] [0-5] (B,152)

        with tf.variable_scope("denses_in_conv_and_mlp", reuse=tf.AUTO_REUSE):# bug!!!
            for _ in range(num_hidden_layers-1):
                print("in conv and mlp layer {}".format(_))
                out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)
            out = tf.layers.dense(out, latent_dim, activation=tf.nn.leaky_relu)

        return out

    return network_fn



@register("conv_and_mlp")
def conv_network_fn(**kwargs):#
    """Used to register custom network type used by Baselines for Overcooked"""

    if "network_kwargs" in kwargs.keys():
        params = kwargs["network_kwargs"]
    else:
        params = kwargs
    if 'TRAIN_MODE' in params.keys():
        train_mode=params['TRAIN_MODE']

    num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
    size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
    num_filters = params["NUM_FILTERS"]
    num_convs = params["NUM_CONV_LAYERS"]
    if "LATENT_DIM" in params.keys():
        latent_dim=params["LATENT_DIM"]
    # BN=params['BN']

    def network_fn(X, latent=None):
        print("conv_and_mlp.shape", X.shape)

        conv_out = tf.layers.conv2d(
            inputs=X,
            filters=num_filters,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.leaky_relu,
            name="conv_initial"
        )

        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"

            conv_out = tf.layers.conv2d(
                inputs=conv_out,
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i)
            )

        out = tf.layers.flatten(conv_out)
        print("out.shape:", out.shape)  # (2000,150)


        print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        if latent != None:


            out = tf.concat([out, latent], axis=1)
        with tf.variable_scope("denses_in_conv_and_mlp", reuse=tf.AUTO_REUSE):# bug!!!
            for _ in range(num_hidden_layers):
                print("in conv and mlp layer {}".format(_))
                out = tf.layers.dense(out, size_hidden_layers, activation=tf.nn.leaky_relu)


        return out

    return network_fn

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


@register("mlp")
def mlp(num_layers=2, num_hidden=64, activation=tf.nn.tanh, layer_norm=False, **other):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        print("in mlp")
        h = tf.layers.flatten(X)
        print("h.shape",h.shape)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
        return h

    return network_fn


@register("human_pre")
def human_pre(num_layers=2, num_hidden=64, activation=tf.nn.tanh, layer_norm=False, **other):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        print("in human_pre")
        h = tf.layers.flatten(X)
        print("h.shape",h.shape)
        for i in range(num_layers):
            h = fc(h, 'human_pre{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            if i==num_layers-1:
                h=tf.nn.tanh(h)
            else:
                h = activation(h)
        return h
    return network_fn




@register("human_prob_pre")
def human_prob_pre(num_layers=2, num_hidden=64, activation=tf.nn.relu, last_activate=tf.identity,layer_norm=False, **other):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X):
        print("in human_prob_pre")
        h = tf.layers.flatten(X)
        print("h.shape",h.shape)
        for i in range(num_layers):
            h = fc(h, 'human_pre{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            if i==num_layers-1:
                h=last_activate(h)
            else:
                h = activation(h)
        return h
    return network_fn




@register("cnn")
def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs)
    return network_fn


@register("cnn_small")
def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h
    return network_fn


@register("lstm")
def lstm(nlstm=128, layer_norm=False):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes.

    Specifically,
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example

    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """

    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


@register("cnn_lstm")
def cnn_lstm(nlstm=128, layer_norm=False, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0]
        nsteps = nbatch // nenv

        h = nature_cnn(X, **conv_kwargs)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)

        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


@register("cnn_lnlstm")
def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)


@register("conv_only")
def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    '''
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer.

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer

    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms


def get_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        print("get_network_bulider")
        return name
    elif name in mapping:

        print("get_network_bulider1")
        print(name)
        print(mapping[name])
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
