import tensorflow as tf
class luo:
    def __init__(self, b, d):
        self.batch_size = b
        self.code_dim = d
    def txt_encoder(self,  txt_vec):
        with tf.variable_scope('txt_encoder'):
            mid = tf.layers.dense(txt_vec, 1024)
            out = tf.tanh(tf.layers.dense(mid, self.code_dim))
            return out
    def image_encoder(self, image_vec):
        with tf.variable_scope('image_encoder'):
            mid = tf.layers.dense(image_vec, 1024)
            out = tf.tanh(tf.layers.dense(mid, self.code_dim))
            return out
    def image_decoder(self, image_embedding, image_dim):
        with tf.variable_scope('image_decoder'):
            mid = tf.layers.dense(image_embedding, 1024)
            out = 3*tf.tanh(tf.layers.dense(mid, image_dim))
            return out
    def image_decoder_text(self, image_embedding, txt_dim):
        with tf.variable_scope('image_decoder_text'):
            mid = tf.layers.dense(image_embedding, 1024)
            out = tf.sigmoid(tf.layers.dense(mid, txt_dim))
            return out
    def txt_decoder(self, txt_embedding, txt_dim):
        with tf.variable_scope('txt_decoder'):
            mid = tf.layers.dense(txt_embedding, 1024)
            out = tf.nn.sigmoid(tf.layers.dense(mid, txt_dim))
            return out
    def txt_decoder_image(self, txt_embedding, image_dim):
        with tf.variable_scope('txt_decoder_image'):
            mid = tf.layers.dense(txt_embedding, 1024)
            out = 3*tf.tanh(tf.layers.dense(mid, image_dim))
            return out
    def disc(self, image_vec, txt_vec, drop_rate=1, reuse=False):
        with tf.variable_scope('disc', reuse=reuse):
            image_mid = tf.layers.dense(image_vec, 1024)
            image_mid_mid = tf.tanh(tf.layers.dense(image_mid, 512))
            txt_mid = tf.layers.dense(txt_vec, 1024)
            txt_mid_mid = tf.tanh(tf.layers.dense(txt_mid, 512))
            mid = image_mid_mid * txt_mid_mid
            mid = tf.nn.dropout(mid, drop_rate)
            ans = tf.layers.dense(mid, 1)
            return ans


