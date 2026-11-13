import matplotlib.pyplot as plt

def plot_forecast(
    targets,
    predictions,
    num_points=200,
):
    targets = targets[:num_points]
    predictions = predictions[:num_points]

    plt.figure(figsize=(12, 5))
    plt.plot(targets, label="Real")
    plt.plot(predictions, label="Predicción")
    plt.xlabel("Tiempo")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_history(
    history,
):
    epochs = range(1, len(history['train_loss']) + 1)

    # Gráfica de Pérdida (Loss)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Training loss')
    plt.plot(epochs, history['val_loss'], label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfica de Tasa de Aprendizaje (Learning Rate)
    if "learning_rate" in history:
        plt.figure()
        plt.plot(epochs, history["learning_rate"], label='Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()
