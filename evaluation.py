# Code for evaluating model with simple holdout validation

# num_validation_samples = 10000
# np.random.shuffle(data)
# validation_data = data[:num_validation_samples]
# training_data = data[num_validation_samples:]
# model = get_model()
# model.fit(training_data, ...)
# validation score = model.evaluate(validation_data, ...)
# model = get_model()
# model.fit(np.concatenate([training_data, validation_data]), ...)
# test score = model.evaluate(test_data, ...)

# Code for evaluating model with k-fold cross-validation

# k = 3
# num_validation_samples = len(data) // k
# np.random.shuffle(data)
# validation_scores = []
# for fold in range(k):
#     validation_data = data[num_validation_samples * fold:
#                            num_validation_samples * (fold + 1)]
#     training_data = np.concatenate(
#         data[:num_validation_samples * fold],
#         data[num_validation_samples * (fold + 1):])
#     model = get_model()
#     model.fit(training_data, ...)
#     validation_score = model.evaluate(validation_data, ...)
#     validation_scores.append(validation_score)
# validation_score = np.average(validation_scores)
# model = get_model()
# model.fit(data, ...)
# test score = model.evaluate(test_data, ...)

# Better code for evaluating model with k-fold cross-validation - shuffle before splitting, multiple iterations