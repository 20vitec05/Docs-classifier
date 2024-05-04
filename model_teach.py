from model import model,train_dataset, val_dataset, test_dataset, epochs, class_weights


model.fit(train_dataset, validation_data= val_dataset, epochs= epochs, class_weight=class_weights)
model.save_weights('classifier_weights.weights.h5')
model.evaluate(test_dataset)