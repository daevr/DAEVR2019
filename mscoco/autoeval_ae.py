import tensorflow as tf
import numpy as np
def local(neb_i, neb_t, is_old=False):
    import numpy as np
    k_text = neb_t
    k_image= neb_i
    vgg_features = np.load('feature.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    bow = np.load('bow.npy')
    data_list = np.arange(len(bow))
    if is_old:
        train_data_list = np.load('train_data_list.npy')
        val_data_list = np.load('val_data_list.npy')
        test_data_list = np.load('test_data_list.npy')
    else:
        np.random.shuffle(data_list)
        train_data_list = data_list[0:10000]
        test_data_list = data_list[10000:15000]
        val_data_list = data_list[15000:20000]
        np.save('train_data_list.npy', train_data_list)
        np.save('test_data_list.npy', test_data_list)
        np.save('val_data_list.npy', val_data_list)

    train_vgg = vgg_features[train_data_list]
    train_bow = bow[train_data_list]

    norm = np.linalg.norm(train_vgg, axis=1)
    dist_new = np.zeros([len(train_data_list), len(train_data_list)], np.float32)
    for i in range(len(train_data_list)):
        dist_new[i] = -np.sum(train_vgg[i] * train_vgg, 1) / norm / \
                      norm[i]
    vgg_features_neb = np.zeros([len(vgg_features), k_image], np.int32)
    for i in range(len(train_data_list)):
        neb = np.argpartition(dist_new[i], k_image)[0:k_image]
        vgg_features_neb[train_data_list[i]] = train_data_list[neb]
    np.save('vgg_feature_neb.npy', vgg_features_neb)

    norm = np.linalg.norm(train_bow, axis=1)
    dist_new = np.zeros([len(train_data_list), len(train_data_list)], np.float32)
    for i in range(len(train_data_list)):
        dist_new[i] = -np.sum(train_bow[i] * train_bow, 1) / norm / \
                      norm[i]
    bow_neb = np.zeros([len(bow), k_text], np.int32)
    for i in range(len(train_data_list)):
        neb = np.argpartition(dist_new[i], k_text)[0:k_text]
        bow_neb[train_data_list[i]] = train_data_list[neb]
    np.save('bow_neb.npy', bow_neb)
    return train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb
def autotrain(train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb):
    import model
    BATCH_SIZE = 128
    LAMBDA = 1
    #data_list = np.load('data_list.npy')
    bow = np.load('bow.npy')
    labels = np.load('label.npy')
    vgg_features = np.load('feature.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    images = tf.placeholder(tf.float32, [None, 4096])
    target_embedding = tf.placeholder(tf.float32, [None, 64])
    # images_target = tf.placeholder(tf.float32, [BATCH_SIZE, 4096])
    images_drop = tf.nn.dropout(images, 0.5)
    texts = tf.placeholder(tf.float32, [None, 2000])
    # texts_target = tf.placeholder(tf.float32, [BATCH_SIZE, 1386])
    texts_drop = tf.nn.dropout(texts, 0.5)
    model = model.luo(BATCH_SIZE, 64)
    text_embedding = model.txt_encoder(texts_drop)
    image_embedding = model.image_encoder(images_drop)
    text_embedding_t = model.txt_encoder(texts, reuse=True)
    image_embedding_t = model.image_encoder(images, reuse=True)
    mixed_embedding = (text_embedding + image_embedding) / 2
    mixed_embedding = tf.nn.l2_normalize(mixed_embedding, 1)

    pre_distance = []
    for i in range(BATCH_SIZE):
        pre_distance.append(tf.reduce_sum(mixed_embedding[i] * mixed_embedding, 1))
    pre_distance = tf.stack(pre_distance)
    mean = tf.reduce_mean(pre_distance)
    var = tf.reduce_mean(tf.square(pre_distance - mean))
    var_loss = tf.square(tf.clip_by_value(0.04- var, 0, 0.04))
    re_image_text = model.txt_decoder_image(text_embedding, 4096)
    re_text_image = model.image_decoder_text(image_embedding, 2000)
    re_image_text_drop = tf.nn.dropout(re_image_text, 0.5)
    re_text_image_drop = tf.nn.dropout(re_text_image, 0.5)
    # disc_real = model.disc(images, texts)
    disc_fake_image = model.disc(re_image_text_drop, texts, reuse=False , drop_rate=1.0)
    disc_fake_text = model.disc(images, re_text_image_drop, reuse=True, drop_rate=1.0)
    loss_image = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image, labels=tf.ones_like(disc_fake_image)))
    loss_text = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_text, labels=tf.ones_like(disc_fake_text)))
    gen_cost = loss_image + loss_text
    consistency_loss = tf.reduce_mean(tf.square(text_embedding - image_embedding))
    vars = tf.trainable_variables()
    gen_params = [v for v in vars if 'coder' in v.name]
    disc_params = [v for v in vars if 'disc' in v.name]
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(
        consistency_loss + gen_cost + 500 * var_loss, var_list=gen_params)

    def test():
        mAP_n = 10000
        P_n = 1000

        def check(x, y):
            flag = bool(np.sum(np.logical_and(x, y)))
            return flag

        def countMAP(result, train_label, po):
            AP = 0
            total_relevant = 0
            buffer_yes = np.zeros(mAP_n)
            Ns = np.arange(1, mAP_n + 1, 1)
            for i in range(mAP_n):
                if check(train_label[result[i]], po):
                    buffer_yes[i] = 1
                    total_relevant += 1

            P = np.cumsum(buffer_yes) / Ns
            if sum(buffer_yes) != 0:
                AP += sum(P * buffer_yes) / sum(buffer_yes)
            return AP

        def countP(result, train_label, po):
            P = 0
            for i in range(P_n):
                if check(train_label[result[i]], po):
                    P += 1
            return P / P_n

        def eval_total(binary_codes_train, binary_codes_test, train_label, test_label):
            lens1 = len(binary_codes_test)
            lens2 = len(binary_codes_train)
            P = 0
            mAP = 0
            dist = np.zeros((lens1, lens2), dtype=np.float32)
            binary_codes_train_norm = np.linalg.norm(binary_codes_train, axis=1)
            binary_codes_test_norm = np.linalg.norm(binary_codes_test, axis=1)
            for i in range(lens1):
                dist[i] = -np.sum(binary_codes_test[i] * binary_codes_train, 1) / binary_codes_train_norm / \
                          binary_codes_test_norm[i]
            results = []
            for i in range(lens1):
                results.append(np.argsort(dist[i]))
            for i in range(lens1):
                mAP += countMAP(results[i], train_label, test_label[i])
                P += countP(results[i], train_label, test_label[i])
            return P / lens1, mAP / lens1

        total_train_embedding = []
        total_test_embedding = []
        for i in range(int(len(train_data_list) / 1000)):
            train_images = vgg_features[train_data_list[i * 1000:min((i + 1) * 1000, len(train_data_list))]]
            _image_embedding = session.run(image_embedding_t, feed_dict={images: train_images})
            total_train_embedding.append(_image_embedding)
        total_train_embedding = np.concatenate(total_train_embedding, axis=0)
        for i in range(int(len(val_data_list) / 1000)):
            test_texts = bow[val_data_list[i * 1000:min((i + 1) * 1000, len(val_data_list))]]
            _text_embedding = session.run(text_embedding_t, feed_dict={texts: test_texts})
            total_test_embedding.append(_text_embedding)
        total_test_embedding = np.concatenate(total_test_embedding, axis=0)
        P, mAP = eval_total(total_train_embedding, total_test_embedding, labels[train_data_list],
                            labels[val_data_list])
        print('mAP')
        print(mAP)
        print('P')
        print(P)
        return mAP

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver2 = tf.train.Saver(disc_params)
        saver2.restore(session, 'disc_block')
        old_map = 0
        for i in range(50001):
            train_list = np.random.randint(0, len(train_data_list), BATCH_SIZE)
            train_images = vgg_features[train_data_list[train_list]]
            # _target_embedding = np.random.uniform(-1, 1, [BATCH_SIZE, 64])
            changer = np.random.randint(0, 2, [BATCH_SIZE, 64], np.int8)
            _target_embedding = np.clip(
                changer * np.random.normal(0.7, 0.2, [BATCH_SIZE, 64]) + (1 - changer) * np.random.normal(-0.7, 0.2,
                                                                                                          [BATCH_SIZE,
                                                                                                           64]), -1, 1)
            train_texts = bow[train_data_list[train_list]]
            # train_images_list = np.zeros_like(train_list, np.int32)
            session.run(gen_train_op,
                        feed_dict={images: train_images, texts: train_texts, target_embedding: _target_embedding})
            #session.run(disc_train_op,
            #           feed_dict={images: train_images, texts: train_texts, target_embedding: _target_embedding})
            if (i % 2000 == 0 and i >= 30000):
                new_map = test()
                if new_map > old_map:
                    old_map = new_map
                    saver.save(session, './referee_advKNN')
            if (i % 100 == 0):
                _closs, _gloss, _var, _t, _i, _d_loss = session.run(
                    [consistency_loss, gen_cost, var, loss_text, loss_image, var_loss],
                    feed_dict={images: train_images, texts: train_texts, target_embedding: _target_embedding})
                print("consistency_loss:%f, gen_loss:%f, var:%f, var_loss:%f" % (_closs, _gloss, _var, _d_loss))
                print('image:%f, text:%f' % (_i, _t))
        print('best map')
        print(old_map)
        session.close()
        return old_map

def binary_train(code_len, rounds):
    import model
    BATCH_SIZE = 128
    tf_idf = np.load('bow.npy')
    vgg_features = np.load('feature.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    images = tf.placeholder(tf.float32, [None, 4096])
    texts = tf.placeholder(tf.float32, [None, 2000])
    embeddings = tf.placeholder(tf.float32, [BATCH_SIZE, 64])
    dist = tf.placeholder(tf.float32, [BATCH_SIZE, BATCH_SIZE])
    model = model.luo(BATCH_SIZE, 64)
    text_embedding = model.txt_encoder(texts)
    image_embedding = model.image_encoder(images)
    embeddings_drop = tf.nn.dropout(embeddings, 0.5)
    binary_vec_drop = model.binary_layer(embeddings_drop, code_len, rate=0.5)
    loss = 0
    for i in range(BATCH_SIZE):
        loss += tf.reduce_mean(tf.square(dist[i, :] - tf.reduce_mean(binary_vec_drop[i] * binary_vec_drop, 1)))
    loss /= BATCH_SIZE
    lossb = tf.square(1 - tf.reduce_mean(binary_vec_drop * binary_vec_drop))
    vars = tf.trainable_variables()
    b_params = [v for v in vars if 'binary' in v.name]
    o_params = [v for v in vars if 'binary' not in v.name]
    b_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, var_list=b_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_data_list = np.load('train_data_list.npy')
    total_train_embedding_images = []
    total_train_embedding_texts = []

    with tf.Session(config=config) as session:
        saver_old = tf.train.Saver(o_params)
        saver = tf.train.Saver()
        session.run(tf.initialize_all_variables())
        saver_old.restore(session, './referee_advKNN')
        for i in range(int(len(train_data_list) / 1000)):
            train_images = vgg_features[train_data_list[i * 1000:min((i + 1) * 1000, len(train_data_list))]]
            _image_embedding = session.run(image_embedding, feed_dict={images: train_images})
            total_train_embedding_images.append(_image_embedding)
        total_train_embedding_images = np.concatenate(total_train_embedding_images, axis=0)
        for i in range(int(len(train_data_list) / 1000)):
            train_texts = tf_idf[train_data_list[i * 1000:min((i + 1) * 1000, len(train_data_list))]]
            _text_embedding = session.run(text_embedding, feed_dict={texts: train_texts})
            total_train_embedding_texts.append(_text_embedding)
        total_train_embedding_texts = np.concatenate(total_train_embedding_texts, axis=0)

        total_train_embedding = (total_train_embedding_texts + total_train_embedding_images) / 2
        total_dist = np.zeros((len(train_data_list), len(train_data_list)), dtype=np.float32)
        norm = np.linalg.norm(total_train_embedding, axis=1)
        for i in range(len(train_data_list)):
            total_dist[i, i:] = np.sum(total_train_embedding[i] * total_train_embedding[i:], 1) / norm[i:] / norm[i]
            total_dist[i:, i] = total_dist[i, i:]
        tempstd2 = total_dist.std()
        total_dist = np.clip((-2/np.mean(np.min(total_dist-1, 1))) * (total_dist - 1) + 1, -1, 1)
        tempstd = total_dist.std()
        for i in range(rounds):
            train_list = np.random.randint(0, len(train_data_list), BATCH_SIZE)
            train_embeddings = total_train_embedding[train_list]
            train_dist = np.zeros([BATCH_SIZE, BATCH_SIZE], np.float32)
            for k in range(BATCH_SIZE):
                train_dist[k] = total_dist[train_list[k]][train_list]
            session.run(b_train_op, feed_dict={embeddings: train_embeddings, dist: train_dist})
            if (i % 10000 == 0):
                saver.save(session, './referee_advKNNB')
            if (i % 100 == 0):
                _loss, _lossB = session.run([loss, lossb], feed_dict={embeddings: train_embeddings, dist: train_dist})
                print(_loss)
                print(_lossB)
        session.close()
        return tempstd, tempstd2
def embedding():
    import model
    mAP_n = 10000
    P_n = 1000

    def HammingDistance(a, b):

        # c = np.logical_xor((np.sign(a)+1).astype(np.bool_),(np.sign(b)+1).astype(np.bool_))
        # dis = np.sum(c.astype(np.int32))
        dis = np.sum(np.square(a - b))
        return dis

    def check(x, y):
        flag = bool(np.sum(np.logical_and(x, y)))
        return flag

    def countMAP(result, train_label, po):
        AP = 0
        total_relevant = 0
        buffer_yes = np.zeros(mAP_n)
        Ns = np.arange(1, mAP_n + 1, 1)
        for i in range(mAP_n):
            if check(train_label[result[i]], po):
                buffer_yes[i] = 1
                total_relevant += 1

        P = np.cumsum(buffer_yes) / Ns
        if sum(buffer_yes) != 0:
            AP += sum(P * buffer_yes) / sum(buffer_yes)
        return AP

    def countP(result, train_label, po):
        P = 0
        for i in range(P_n):
            if check(train_label[result[i]], po):
                P += 1
        return P / P_n

    def eval_total(binary_codes_train, binary_codes_test, train_label, test_label):
        lens1 = len(binary_codes_test)
        lens2 = len(binary_codes_train)
        P = 0
        mAP = 0
        dist = np.zeros((lens1, lens2), dtype=np.float32)
        binary_codes_train_norm = np.linalg.norm(binary_codes_train, axis=1)
        binary_codes_test_norm = np.linalg.norm(binary_codes_test, axis=1)
        for i in range(lens1):
            dist[i] = -np.sum(binary_codes_test[i] * binary_codes_train, 1) / binary_codes_train_norm / \
                      binary_codes_test_norm[i]
        results = []
        for i in range(lens1):
            results.append(np.argsort(dist[i]))
        for i in range(lens1):
            mAP += countMAP(results[i], train_label, test_label[i])
            P += countP(results[i], train_label, test_label[i])
        return P / lens1, mAP / lens1

    BATCH_SIZE = 128
    bow = np.load('bow.npy')
    labels = np.load('label.npy')
    vgg_features = np.load('feature.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    images = tf.placeholder(tf.float32, [None, 4096])
    texts = tf.placeholder(tf.float32, [None, 2000])
    model = model.luo(BATCH_SIZE, 64)
    text_embedding = model.txt_encoder(texts)
    image_embedding = model.image_encoder(images)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_data_list = np.load('train_data_list.npy')
    test_data_list = np.load('test_data_list.npy')
    total_train_embedding = []
    total_test_embedding = []
    with tf.Session(config=config) as session:
        saver = tf.train.Saver()
        saver.restore(session, './referee_advKNN')
        for i in range(int(len(train_data_list) / 1000)):
            train_images = vgg_features[train_data_list[i * 1000:min((i + 1) * 1000, len(train_data_list))]]
            _image_embedding = session.run(image_embedding, feed_dict={images: train_images})
            total_train_embedding.append(_image_embedding)
        total_train_embedding = np.concatenate(total_train_embedding, axis=0)
        for i in range(int(len(test_data_list) / 1000)):
            test_texts = bow[test_data_list[i * 1000:min((i + 1) * 1000, len(val_data_list))]]
            _text_embedding = session.run(text_embedding, feed_dict={texts: test_texts})
            total_test_embedding.append(_text_embedding)
        total_test_embedding = np.concatenate(total_test_embedding, axis=0)
        session.close()
    P, mAP = eval_total(total_train_embedding, total_test_embedding, labels[train_data_list], labels[test_data_list])
    print('mAP')
    print(mAP)
    print('P')
    print(P)
    return P, mAP
def embeddingB(code_len):
    import model
    mAP_n = 10000
    P_n = 1000

    def HammingDistance(a, b):

        # c = np.logical_xor((np.sign(a)+1).astype(np.bool_),(np.sign(b)+1).astype(np.bool_))
        # dis = np.sum(c.astype(np.int32))
        dis = np.sum(np.square(a - b))
        return dis

    def check(x, y):
        flag = bool(np.sum(np.logical_and(x, y)))
        return flag

    def countMAP(result, train_label, po):
        AP = 0
        total_relevant = 0
        buffer_yes = np.zeros(mAP_n)
        Ns = np.arange(1, mAP_n + 1, 1)
        for i in range(mAP_n):
            if check(train_label[result[i]], po):
                buffer_yes[i] = 1
                total_relevant += 1

        P = np.cumsum(buffer_yes) / Ns
        if sum(buffer_yes) != 0:
            AP += sum(P * buffer_yes) / sum(buffer_yes)
        return AP

    def countP(result, train_label, po):
        P = 0
        for i in range(P_n):
            if check(train_label[result[i]], po):
                P += 1
        return P / P_n

    def eval_total(binary_codes_train, binary_codes_test, train_label, test_label):
        lens1 = len(binary_codes_test)
        lens2 = len(binary_codes_train)
        P = 0
        mAP = 0
        dist = np.zeros((lens1, lens2), dtype=np.float32)
        binary_codes_train_norm = np.linalg.norm(binary_codes_train, axis=1)
        binary_codes_test_norm = np.linalg.norm(binary_codes_test, axis=1)
        for i in range(lens1):
            dist[i] = -np.sum(binary_codes_test[i] * binary_codes_train, 1) / binary_codes_train_norm / \
                      binary_codes_test_norm[i]
        results = []
        for i in range(lens1):
            results.append(np.argsort(dist[i]))
        for i in range(lens1):
            mAP += countMAP(results[i], train_label, test_label[i])
            P += countP(results[i], train_label, test_label[i])
        return P / lens1, mAP / lens1

    BATCH_SIZE = 128
    bow = np.load('bow.npy')
    labels = np.load('label.npy')
    vgg_features = np.load('feature.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    images = tf.placeholder(tf.float32, [None, 4096])
    texts = tf.placeholder(tf.float32, [None, 2000])
    model = model.luo(BATCH_SIZE, 64)
    text_embedding = model.txt_encoder(texts)
    text_embedding = tf.sign(model.binary_layer(text_embedding, code_len))
    image_embedding = model.image_encoder(images)
    image_embedding = tf.sign(model.binary_layer(image_embedding, code_len, reuse=True))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_data_list = np.load('train_data_list.npy')
    test_data_list = np.load('test_data_list.npy')
    total_train_embedding = []
    total_test_embedding = []
    with tf.Session(config=config) as session:
        saver = tf.train.Saver()
        saver.restore(session, './referee_advKNNB')
        for i in range(int(len(train_data_list) / 1000)):
            train_images = vgg_features[train_data_list[i * 1000:min((i + 1) * 1000, len(train_data_list))]]
            _image_embedding = session.run(image_embedding, feed_dict={images: train_images})
            total_train_embedding.append(_image_embedding)
        total_train_embedding = np.concatenate(total_train_embedding, axis=0)
        for i in range(int(len(test_data_list) / 1000)):
            test_texts = bow[test_data_list[i * 1000:min((i + 1) * 1000, len(test_data_list))]]
            _text_embedding = session.run(text_embedding, feed_dict={texts: test_texts})
            total_test_embedding.append(_text_embedding)
        total_test_embedding = np.concatenate(total_test_embedding, axis=0)
        session.close()
    P, mAP = eval_total(total_train_embedding, total_test_embedding, labels[train_data_list], labels[test_data_list])
    print('mAP')
    print(mAP)
    print('P')
    print(P)
    return P, mAP

def disc_train(train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb, neb_i, neb_t, rounds):
    import model
    BATCH_SIZE = 256
    bow = np.load('bow.npy')
    bow_neb = np.load('bow_neb.npy')
    vgg_features = np.load('feature.npy')
    vgg_features_neb = np.load('vgg_feature_neb.npy')
    vgg_features = np.tanh((vgg_features) / np.std(vgg_features))
    model = model.luo(BATCH_SIZE, 64)
    images = tf.placeholder(tf.float32, [None, 4096])
    texts = tf.placeholder(tf.float32, [None, 2000])
    images_drop = tf.nn.dropout(images, 0.5)
    texts_drop = tf.nn.dropout(texts, 0.5)
    labels = tf.placeholder(tf.float32, [None, 1])
    disc_logits = model.disc(images_drop, texts_drop, drop_rate=0.5)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        for i in range(rounds):
            train_list = train_data_list[np.random.randint(0, len(train_data_list), 2 * BATCH_SIZE)]
            ne_train_list = train_list[128:]
            train_list = train_list[0:128]
            train_image_list = []
            for j in range(len(train_list)):
                ram = np.random.randint(0, neb_i)
                train_image_list.append(vgg_features_neb[train_list[j]][ram])
            train_text_list = []
            for j in range(len(train_list)):
                ram = np.random.randint(0, neb_t)
                train_text_list.append(bow_neb[train_list[j]][ram])
            related_list = set()
            for j in range(len(train_list)):
                for x in bow_neb[train_list[j]]:
                    related_list.add(x)
                for x in vgg_features_neb[train_list[j]]:
                    related_list.add(x)

            ne_train_list = [w for w in ne_train_list if w not in related_list]
            ne_train_list = ne_train_list[0:128]
            ne_train_texts = bow[ne_train_list]
            train_images = vgg_features[train_image_list]
            train_images = np.concatenate([train_images, train_images])
            train_texts = bow[train_text_list]
            train_texts = np.concatenate([train_texts, ne_train_texts])
            train_labels = np.concatenate([np.ones([128, 1], np.float32), np.zeros([128, 1], np.float32)])
            session.run(train_op, feed_dict={images: train_images, texts: train_texts, labels: train_labels})
            if i % 100 == 0:
                _loss = session.run(loss, feed_dict={images: train_images, texts: train_texts, labels: train_labels})
                print(_loss)
            if i % 5000 == 0:
                saver.save(session, './disc_block')
        session.close()

if __name__=="__main__":
    mAP =[]
    mAPB64 = []
    mAPB32 = []
    mAPB16 = []
    std = []
    std2 = []
    bmaps = []
    for i in range(5):
        train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb = local(23, 14)
        tf.reset_default_graph()
        disc_train(train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb, 23, 14, 20000)
        tf.reset_default_graph()
        bmap = autotrain(train_data_list, val_data_list, test_data_list, vgg_features_neb, bow_neb)
        bmaps.append(bmap)
        tf.reset_default_graph()
        _, t = embedding()
        mAP.append(t)
        rounds = 80000
        tf.reset_default_graph()
        tstd, tstd2 = binary_train(64, rounds)
        tf.reset_default_graph()
        _, t = embeddingB(64)
        mAPB64.append(t)
        tf.reset_default_graph()
        _ = binary_train(32, rounds)
        tf.reset_default_graph()
        _, t = embeddingB(32)
        mAPB32.append(t)
        tf.reset_default_graph()
        _ = binary_train(16, rounds)
        tf.reset_default_graph()
        _, t = embeddingB(16)
        mAPB16.append(t)

    print(tstd)
    print(tstd2)
    std.append(tstd)
    std2.append(tstd2)
    print(bmaps)
    print(mAP)
    print(mAPB64)
    print(mAPB32)
    print(mAPB16)
    print(std)
    print(std2)