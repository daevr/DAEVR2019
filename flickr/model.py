import tensorflow as tf
class luo:
    def __init__(self, b, d):
        self.batch_size = b
        self.code_dim = d
    def txt_encoder(self,  txt_vec, reuse=False):
        with tf.variable_scope('txt_encoder', reuse=reuse):
            layer1 = tf.tanh(tf.layers.dense(txt_vec, 1024))
            layer2 = tf.tanh(tf.layers.dense(layer1, 1024))
            out = tf.tanh(tf.layers.dense(layer2, self.code_dim))
            return out
    def image_encoder(self, image_vec, reuse=False):
        with tf.variable_scope('image_encoder', reuse=reuse):
            layer1 = tf.tanh(tf.layers.dense(image_vec, 1024))
            layer2 = tf.tanh(tf.layers.dense(layer1, 1024))
            out = tf.tanh(tf.layers.dense(layer2, self.code_dim))
            return out
    def image_decoder(self, image_embedding, image_dim):
        with tf.variable_scope('image_decoder'):
            mid = tf.tanh(tf.layers.dense(image_embedding, 1024))
            out = tf.tanh(tf.layers.dense(mid, image_dim))
            return out
    def image_decoder_text(self, image_embedding, txt_dim):
        with tf.variable_scope('image_decoder_text'):
            layer1 = tf.layers.dense(image_embedding, 1024)
            layer2 = tf.layers.dense(layer1, 1024)
            out = tf.sigmoid(tf.layers.dense(layer2, txt_dim))
            return out
    def txt_decoder(self, txt_embedding, txt_dim, reuse=False):
        with tf.variable_scope('txt_decoder'):
            mid = tf.tanh(tf.layers.dense(txt_embedding, 1024))
            out = tf.sigmoid(tf.layers.dense(mid, txt_dim))
            return out
    def txt_decoder_image(self, txt_embedding, image_dim, reuse=False):
        with tf.variable_scope('txt_decoder_image'):
            layer1 = tf.layers.dense(txt_embedding, 1024)
            layer2 = tf.layers.dense(layer1, 1024)
            out = tf.tanh(tf.layers.dense(layer2, image_dim))
            return out
    def disc(self, image_vec, txt_vec, drop_rate=0.5, reuse=False):
        with tf.variable_scope('disc', reuse=reuse):
            image_mid = tf.tanh(tf.layers.dense(image_vec, 1024))
            image_mid2 = tf.tanh(tf.layers.dense(image_mid, 1024))
            txt_mid = tf.tanh(tf.layers.dense(txt_vec, 1024))
            txt_mid2 = tf.tanh(tf.layers.dense(txt_mid, 1024))
            mid = image_mid2 * txt_mid2
            mid = tf.nn.dropout(mid, drop_rate)
            ans = tf.layers.dense(mid, 1)
            return ans
    def disc_distribution(self, distribution, reuse=False):
        with tf.variable_scope('disc_distribution', reuse=reuse):
            layer1 = tf.tanh(tf.layers.dense(distribution, 1024))
            layer2 = tf.tanh(tf.layers.dense(layer1, 1024))
            ans = tf.layers.dense(layer2, 1)
            return ans
    def binary_layer(self, embedding, dim, reuse=False):
        with tf.variable_scope('binary', reuse=reuse):
            layer1 = tf.tanh(tf.layers.dense(embedding, 1024))
            layer2 = tf.tanh(tf.layers.dense(layer1, 1024))
            out = tf.tanh(tf.layers.dense(layer2, dim))
            return out


