nworkers=2
#lr=0.1
lr=0.01
batch_size=64
#dnn=resnet20
#dnn=vgg16
#dnn=alexnet
#dnn=resnet50
#dataset=imagenet
#max_epochs=180
max_epochs=95
#data_dir=./data
#data_dir=/home/comp/csshshi/data/imagenet/ILSVRC2012_dataset
data_dir=/home/shshi/data/imagenet/imagenet_hdf5
mpirun -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x NCCL_DEBUG=DEBUG -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python torch_mnist.py
