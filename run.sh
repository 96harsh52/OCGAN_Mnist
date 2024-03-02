python ocganTrain.py\
 --batchsize 128\
 --isize 32\
 --nc 1\
 --extralayers 0\
 --device 'gpu'\
 --anomaly_class 0\
 --resume ''\
 --niter 30\
 --lr 0.0002\
 



 # --batchsize 32: Set the batch size for training data to 32.
# --isize 32: Set the size of input images to 32x32 pixels.
# --nc 1: Set the number of input channels to 1 (assuming grayscale images).
# --extralayers 0: No extra layers beyond the default architecture.
# --device 'gpu': Utilize the GPU for training if available.
# --anomaly_class 0: Define the anomaly class (0 in this case, assuming single-class anomaly detection).
# --resume '': No resume training from a previous checkpoint.
# --niter 30: Set the number of training epochs to 30.
# --lr 0.0002: Set the learning rate to 0.0002.
# --outf './output': Define the output directory for saving training results.