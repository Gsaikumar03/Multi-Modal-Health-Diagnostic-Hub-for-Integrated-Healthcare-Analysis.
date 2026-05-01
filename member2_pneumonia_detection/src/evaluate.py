def evaluate_model(model, test_ds):

    y_true = []
    y_prob = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_prob.extend(preds.ravel())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    return y_true, y_prob