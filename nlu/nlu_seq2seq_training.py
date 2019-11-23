import matplotlib.pyplot as plt


def plot_training_accuracy(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    # plt.savefig('plot.png')

# EPOCHS = 50
# history = model.fit([input_data_train, teacher_data_train], target_data_train, batch_size=BATCH_SIZE,
#     epochs=EPOCS,
#     validation_data=([input_data_test, teacher_data_test], target_data_test))
# plot_training_accuracy(history)