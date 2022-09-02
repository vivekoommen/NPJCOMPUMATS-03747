import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
import time
import matplotlib.pyplot as plt

from ae import ae

@tf.function(jit_compile=True)
def train(model, x, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss   = model.loss(y_pred, x)[0]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return(loss)

def main():
    np.random.seed(23)

    #Load dataset
    d = np.load('data/train_128_128.npz')
    x_train = d['X_func']
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], x_train.shape[2], 1))
    shuffler = np.random.permutation(len(x_train))
    x_train = x_train[shuffler]
    x_train = np.reshape(x_train , (-1, 100, 128, 128, 1))
    #x_train = x_train[:, 10:]
    x_train = np.reshape(x_train, (-1,128,128,1))
    num_samples = x_train.shape[0]

    Par={}

    x_train = (x_train - np.min(x_train))/(np.max(x_train)-np.min(x_train))


    print('x_train shape: ', x_train.shape)

    d = np.load('data/test_128_128.npz')
    x_test = d['X_func']

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], x_test.shape[2], 1))
    shuffler = np.random.permutation(len(x_test))
    x_test = x_test[shuffler]
    x_test = np.reshape(x_test , (-1, 100, 128, 128, 1))
    x_test = np.reshape(x_test, (-1,128,128,1))

    x_test = (x_test - np.min(x_test))/(np.max(x_test) - np.min(x_test))

    print('x_test shape:  ',x_test.shape )

    address = 'saved_models/ae_models'
    Par['address'] = address

    #Create an object
    model = ae()
    print('Model created')

    n_epochs = 12
    batch_size = 1
    n_batches = int(num_samples/batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-4)

    begin_time = time.time()
    print('Training Begins')
    for i in range(n_epochs+1):
        for j in np.arange(0, num_samples-batch_size, batch_size):
            loss = train(model, x_train[j:(j+batch_size)], optimizer)

        if i%1 == 0:

            model.save_weights(address + "/model_"+str(i))

            train_loss = loss.numpy()

            y_pred = model(x_test)
            val_loss = np.mean( (y_pred - x_test)**2 )

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            model.index_list.append(i)
            model.train_loss_list.append(train_loss)
            model.val_loss_list.append(val_loss)

    print('Training complete')

    #Convergence plot
    index_list = model.index_list
    train_loss_list = model.train_loss_list
    val_loss_list = model.val_loss_list
    np.savez(address+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig( address + "/convergence.png", dpi=800)
    plt.close()

    best_model_number = index_list[np.argmin(val_loss_list)]
    print('Best autencoder model: ', best_model_number)

    np.save(address+'/best_ae_model_number', best_model_number)

    print('--------Complete--------')

main()
