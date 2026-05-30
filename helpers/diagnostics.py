

def top5_accuracy(true_labels, top5_predictions):
    correct = 0
    for true, top5 in zip(true_labels, top5_predictions):
        if true in top5:
            correct += 1
    return correct / len(true_labels)