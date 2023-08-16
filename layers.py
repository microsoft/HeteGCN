import tensorflow as tf 

def dropout(x, dropout, seed=0):
    """
    This function takes a tensor (dense/sparse) as input, applies dropout. 
    """
    if isinstance(x, tf.SparseTensor):
        values = x.values 
        values = tf.nn.dropout(values, keep_prob=1-dropout, seed=seed)
        res = tf.SparseTensor(x.indices, values, x.dense_shape)
    else:
        res = tf.nn.dropout(x, keep_prob=1-dropout, seed=seed)
    return res

class GCN_Layer():
    """
    A generic GCN layer implementing the following propagtion steps 
        1. AXW + b (if has_features=True and bias=True)
        2. AXW (if has_features=True and bias=False)
        3. AW (if has_features=False) Equivalent to setting X=I, 
           W is treated as learnable embeddings
    """
    def __init__(self, args):
        self.args = args
        
        self.has_features = 'X' in self.args
        self.dropout = self.args['dropout']
        self.seed = self.args['seed']
        
        self.A = self.args['A']
        self.input_dims = self.args['input_dims']
        self.output_dims = self.args['output_dims']
        self.bias = self.args.get('bias', False)
        self.setup_weights()
        self.propagate()

    def setup_weights(self):
        """
        Setups and initializes the learnable parameters of the layer.
        """        
        self.aux_embeddings = []
        with tf.variable_scope('Aggregator'):
            self.W = tf.get_variable('W', shape=[self.input_dims, self.output_dims], dtype=tf.float32,  initializer=tf.contrib.layers.xavier_initializer())
            
            if self.has_features:
                # X can be a placeholder if used in the first layer
                # Else it is the output of the previous layer
                self.X = self.args['X']
                
                if self.bias:
                    self.b = tf.get_variable('b', shape=[self.output_dims, ], dtype=tf.float32,  initializer=tf.zeros_initializer())
                tf.add_to_collection('weights', self.W)

                #This is used to normalize the regularization losses.
                tf.add_to_collection('#_of_weights', self.input_dims*self.output_dims)
            else:
                tf.add_to_collection('embeddings', self.W)
            
    def propagate(self):
        """
        Propagation logic is defined here.
        """
        if self.has_features: 
            #AXW
            self.W = dropout(self.W, self.dropout, self.seed)
            if isinstance(self.X, tf.SparseTensor):
                self.outputs = tf.sparse_tensor_dense_matmul(self.X, self.W)
                if self.A:
                    self.outputs = tf.sparse_tensor_dense_matmul(self.A, self.outputs)
            else:
                if self.A:
                    self.outputs = tf.sparse_tensor_dense_matmul(self.A, self.X)
                    self.aux_embeddings = self.outputs
                else:
                    self.outputs = self.X
                self.outputs = tf.matmul(self.outputs, self.W)
            
            #AXW (+ b)
            if self.bias:
                self.outputs = tf.add(self.outputs, self.b)
        else:
            #AW (or equivalent to setting X=I)
            self.outputs = tf.sparse_tensor_dense_matmul(self.A, self.W)
        return self.outputs

    def __call__(self):
        return self.outputs
