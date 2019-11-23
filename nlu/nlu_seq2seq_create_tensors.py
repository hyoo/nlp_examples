import tensorflow as tf
import numpy as np


def max_length(tensor):
    return max(len(t) for t in tensor)


def create_tensors(input_tensor, target_tensor, nb_sample=9999999, max_len=0):
    len_input, len_target = max_length(input_tensor), max_length(target_tensor)
    len_input = max(len_input, max_len)
    len_target = max(len_target, max_len)

    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                               maxlen=len_input,
                                                               padding='post')

    teacher_data = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                 maxlen=len_target,
                                                                 padding='post')

    target_data = [[teacher_data[n][i + 1] for i in range(len(teacher_data[n]) - 1)] for n in range(len(teacher_data))]
    target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding='post')
    target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

    nb = len(input_data)
    p = np.random.permutation(nb)
    input_data = input_data[p]
    teacher_data = teacher_data[p]
    target_data = target_data[p]

    return input_data[:min(nb_sample, nb)], teacher_data[:min(nb_sample, nb)], target_data[:min(nb_sample, nb)], \
        len_input, len_target

# input_data_train, teacher_data_train, target_data_train, \
# len_input_train, len_target_train  = create_tensors(input_tensor_train, target_tensor_train)
# input_data_test, teacher_data_test, target_data_test, \
# len_input_test, len_target_test  = create_tensors(input_tensor_test, target_tensor_test, max_len=len_input_train)
