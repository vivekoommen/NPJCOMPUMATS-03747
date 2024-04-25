import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from ae import ae

import matplotlib
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

from don import DeepONet_Model ###############################

def preprocess(x):
    X_func = np.reshape(x, (-1,100, 14,14))
    index = list(range(10,90))
    X_func = X_func[:, list(range(10,90))]
    X_func = np.transpose(X_func, axes=[0, 2, 3, 1])
    print(X_func.shape)

    X_loc = np.array(index)/100
    X_loc = X_loc[:,None]
    print(X_loc.shape)

    y = np.reshape(x, (-1,100, 196))
    y = y[:, index]
    print(y.shape)

    return X_func, X_loc, y

def tensor(x):
    return tf.convert_to_tensor(x, dtype=tf.float32)


# @tf.function(jit_compile=True)
def train(don_model, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.Loss(y_hat, y)[0]

    gradients = tape.gradient(loss, don_model.trainable_variables)
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return(loss)

def error_metric(true, pred):
    #true - [samples, time steps, 128, 128]
    #pred - [samples, time steps, 128, 128]
    pred = np.reshape(pred, (-1,90, 128, 128 ))
    num = np.abs(true - pred)**2 #[samples, time steps, 128, 128]
    num = np.sum(num) #[samples, time steps]
    den = np.abs(true)**2
    den = np.sum(den)

    return num/den

def show_error(don_model, ae_model, X_func, X_loc, pf_true):
    y_pred = don_model(X_func, X_loc)
    y_pred = np.reshape(y_pred, (-1,ae_model.latent_dim))

    pf_pred = ae_model.decode(y_pred)
    error = error_metric(pf_true, pf_pred)
    print('L2 norm of relative error: ', error)

def main():
    Par = {}


    train_dataset = np.load('data/train_ld.npz')['X_func']
    test_dataset = np.load('data/test_ld.npz')['X_func']


    Par['address'] = 'saved_models/don_models' #####################################################

    print(Par['address'])
    print('------\n')


    X_func_train, X_loc_train, y_train = preprocess(train_dataset)
    X_func_test, X_loc_test, y_test = preprocess(test_dataset)
    Par['n_channels'] = X_func_train.shape[-1]

    print('X_func_train: ', X_func_train.shape, '\nX_loc_train: ', X_loc_train.shape, '\ny_train: ', y_train.shape)
    print('X_func_test: ', X_func_test.shape, '\nX_loc_test: ', X_loc_test.shape, '\ny_test: ', y_test.shape)

    Par['mean'] = np.mean(X_func_train)
    Par['std'] =  np.std(X_func_train)

    print('mean: ', Par['mean'])
    print('std : ', Par['std'])

    don_model = DeepONet_Model(Par)
    n_epochs = 12
    batch_size = 1

    print("DeepONet Training Begins")
    begin_time = time.time()




    for i in range(n_epochs+1):
        for end in np.arange(batch_size, X_func_train.shape[0]+1, batch_size):
            start = end - batch_size
            loss = train(don_model, tensor(X_func_train[start:end]), tensor(X_loc_train), tensor(y_train[start:end]))

        if i%1 == 0:

            don_model.save_weights(Par['address'] + "/model_"+str(i) +".weights.h5")

            train_loss = loss.numpy()

            y_hat = don_model(X_func_test, X_loc_test)

            val_loss = np.mean( (y_hat - y_test)**2 )

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)



    #Convergence plot
    index_list = don_model.index_list
    train_loss_list = don_model.train_loss_list
    val_loss_list = don_model.val_loss_list
    np.savez(Par['address']+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)


    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig(Par["address"] + "/convergence.png", dpi=800)
    plt.close()

    if True:
        ae_model = ae()
        ae_model_number = np.load('saved_models/ae_models/best_ae_model_number.npy')
        ae_model_address = "saved_models/ae_models/model_"+str(ae_model_number)+".weights.h5"
        ae_model.load_weights(ae_model_address)


        don_model = DeepONet_Model(Par)
        don_model_number = index_list[np.argmin(val_loss_list)]
        np.save('data/best_don_model_number', don_model_number)
        don_model_address = Par['address'] + "/model_"+str(don_model_number)+".weights.h5"
        don_model.load_weights(don_model_address)

        print('best DeepONet model: ', don_model_number)

        n_samples = 20

        pf_true = np.load('data/train_128_128.npz')['X_func'].astype(np.float32)
        pf_true = (pf_true[:10000] - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1,100,128,128))
        pf_true_train = pf_true[:n_samples, 10:]

        pf_true = np.load('data/test_128_128.npz')['X_func'].astype(np.float32)
        pf_true = (pf_true[:10000] - np.min(pf_true))/(np.max(pf_true) - np.min(pf_true))
        pf_true = np.reshape(pf_true, (-1,100,128,128))
        pf_true_test = pf_true[:n_samples, 10:]

        X_loc = np.linspace(0,1,100)[10:][:,None]

        print('Train Dataset')
        show_error(don_model, ae_model, X_func_train[:n_samples], X_loc, pf_true_train)

        print('Test Dataset')
        show_error(don_model, ae_model, X_func_test[:n_samples], X_loc, pf_true_test)

        print('--------Complete--------')


main()
