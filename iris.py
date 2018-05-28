from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

def init():
    train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url)

    test_url = "http://download.tensorflow.org/data/iris_test.csv"
    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                      origin=test_url)

    train_dataset = tf.data.TextLineDataset(train_dataset_fp)
    train_dataset = train_dataset.skip(1)             # skip the first header row
    train_dataset = train_dataset.map(parse_csv)      # parse each row

    test_dataset = tf.data.TextLineDataset(test_fp)
    test_dataset = test_dataset.skip(1)             # skip header row
    test_dataset = test_dataset.map(parse_csv)      # parse each row
    
    # peanut gallery comments ==================================================
    # 
    # Copied from tutorial:
    # training works best if the examples are in random order. 
    # Use tf.data.Dataset.shuffle to randomize entries, setting buffer_size to a
    # value larger than the number of examples (120 in this case)
    # --------------------------------------------------------------------------
    train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
    test_dataset = test_dataset.shuffle(1000)       # randomize
    
    # peanut gallery comments ==================================================
    # 
    # Copied from tutorial:
    # To train the model faster, the dataset's batch size is set to 32 examples
    # to train at once.
    # --------------------------------------------------------------------------
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)
    
    # peanut gallery comments ==================================================
    # SO MANY FUCKING MAGIC NUMBERS!!!!!
    # 
    # Copied from tutorial:
    # The ideal number of hidden layers and neurons depends on the problem and 
    # the dataset. Like many aspects of machine learning, picking the best shape 
    # of the neural network requires a mixture of knowledge and experimentation. 
    # As a rule of thumb, increasing the number of hidden layers and neurons 
    # typically creates a more powerful model, which requires more data to train 
    # effectivel
    # --------------------------------------------------------------------------
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(3) # 3 === the number of labels
    ])

    # peanut gallery comments ==================================================
    # learning_rate=0.01 OH LOOK MA A MAGIC NUMBER
    # --------------------------------------------------------------------------
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # peanut gallery comments ==================================================
    # CONSTANTLY WITH THE MAGIC NUMBERS, I CAN'T WITH THIS
    # An epoch is a full loop over the dataset (so we'll look over it 201 times)
    # there are 120 examples... so move the 1 to the end and get 201? lolololol
    # magic is magic is magic
    # --------------------------------------------------------------------------
    num_epochs = 180

    for epoch in range(num_epochs):
      epoch_loss_avg = tfe.metrics.Mean()
      epoch_accuracy = tfe.metrics.Accuracy()

      # Training loop - using batches of 32
      # peanut gallery comments ================================================
      # x = features, y = label
      # ------------------------------------------------------------------------
      for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

      # end epoch
      train_loss_results.append(epoch_loss_avg.result())
      train_accuracy_results.append(epoch_accuracy.result())
      
      if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

    # peanut gallery comments ==================================================
    # graphs about loss and accuracy
    # --------------------------------------------------------------------------        
    # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    # fig.suptitle('Training Metrics')

    # axes[0].set_ylabel("Loss", fontsize=14)
    # axes[0].plot(train_loss_results)

    # axes[1].set_ylabel("Accuracy", fontsize=14)
    # axes[1].set_xlabel("Epoch", fontsize=14)
    # axes[1].plot(train_accuracy_results)

    # plt.show()
    
    test_accuracy = tfe.metrics.Accuracy()
    for (x, y) in test_dataset:
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


    class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset)

    for i, logits in enumerate(predictions):
      class_idx = tf.argmax(logits).numpy()
      name = class_ids[class_idx]
      print("Example {} prediction: {}".format(i, name))

def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

init()
