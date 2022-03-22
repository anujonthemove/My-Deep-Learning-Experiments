import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('seaborn-poster')


def plot(history):
    # summarize history for accuracy
    plt.plot(history.history['mae'], '--')
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'], '--')
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()
