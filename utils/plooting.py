import matplotlib.pyplot as plt

def plot_losses(
    history
    ):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(
        epochs,
        history['train_loss'],
        label='Training Loss'
        )
    plt.plot(
        epochs,
        history['val_loss'],
        label='Validation Loss'
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
