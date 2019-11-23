import time
import os
import tensorflow as tf

from nlu_seq2seq_load_data import load_atis
from nlu_seq2seq_vocab_size import get_vocab_size
from nlu_seq2seq_create_tensors import create_tensors
from nlu_seq2seq_attention_encoder import Encoder
from nlu_seq2seq_attention_decoder import Decoder

def run():
    # load raw data
    t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
        input_tensor_train, target_tensor_train, \
        query_data_train, intent_data_train, intent_data_label_train, slot_data_train \
        = load_atis('ms_cntk_atis.train.pkl', verbose=False)

    t2i_test, s2i_test, in2i_test, i2t_test, i2s_test, i2in_test, \
        input_tensor_test, target_tensor_test, \
        query_data_test, intent_data_test, intent_data_label_test, slot_data_test \
        = load_atis('ms_cntk_atis.test.pkl', verbose=False)

    # vocab
    vocab_in_size, vocab_out_size = get_vocab_size(t2i_train, t2i_test, s2i_train, s2i_test)

    # tensors
    input_data_train, teacher_data_train, target_data_train, \
        len_input_train, len_target_train = create_tensors(input_tensor_train, target_tensor_train)
    input_data_test, teacher_data_test, target_data_test, \
        len_input_test, len_target_test = create_tensors(input_tensor_test, target_tensor_test, max_len=len_input_train)

    # data load
    vocab_out_size = 180
    BUFFER_SIZE = len(input_data_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024

    dataset = tf.data.Dataset.from_tensor_slices((input_data_train, teacher_data_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # build encoder
    encoder = Encoder(vocab_in_size, embedding_dim, units, BATCH_SIZE)

    # build decoder
    decoder = Decoder(vocab_out_size, embedding_dim, units, BATCH_SIZE)

    # optimizer & checkpoint
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    checkpoint_dir = './model_dir/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    # training
    @tf.function
    def train_step(inp, targ, enc_hidden, s2i=s2i_train):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            # s2i['BOS'] = 178
            dec_input = tf.expand_dims([178] * BATCH_SIZE, 1)

            # teacher forcing
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    EPOCHS = 20
    epoch_loss = []
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(N_BATCH)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        epoch_loss.append(total_loss / N_BATCH)
        print('Eopch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
        print('Time take for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    run()
