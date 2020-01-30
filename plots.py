from matplotlib import pyplot as plt

def plot_accuracy(history, chance=None, filename=None, ax=None, figsize=(12, 8)):
    # Plot training metrics
        
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = len(history.history['acc'])
    # for metric in history.history:
    #     ax.plot(range(1, epochs+1), history.history[metric], label=metric)
    ax.plot(range(1, epochs+1), history.history['acc'], label="Training")
    if 'val_acc' in history.history:
        ax.plot(range(1, epochs+1), history.history['val_acc'], label="Validation")
    if chance:
        ax.axhline(y=chance, color='grey', linestyle='--', label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, epochs+1))
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0, 1))
    ax.legend()
    if filename:
        fig.savefig(filename)
    return (fig, ax)

def plot_loss(history, filename=None, ax=None, figsize=(12, 8)):
    # Plot training metrics
        
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    epochs = len(history.history['loss'])
    # for metric in history.history:
    #     ax.plot(range(1, epochs+1), history.history[metric], label=metric)
    ax.plot(range(1, epochs+1), history.history['loss'], label="Training")
    if 'val_loss' in history.history:
        ax.plot(range(1, epochs+1), history.history['val_loss'], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_xlim((0, epochs+1))
    ax.set_ylabel("Loss")
    # ax.set_ylim((0, 1))
    ax.legend()
    if filename:
        fig.savefig(filename)
    return (fig, ax)