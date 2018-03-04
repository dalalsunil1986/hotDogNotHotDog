#import Dependencies
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import time 

import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.INFO)
from mxnet.test_utils import download

batch_size = 64
ctx = [mx.cpu()]

train_iter = mx.image.ImageIter(batch_size=batch_size,data_shape=(3,270,270),label_width=1,path_imgrec='mk_train.rec',shuffle=False,part_index=0,num_parts=1)
val_iter = mx.image.ImageIter(batch_size=batch_size,data_shape=(3,270,270),label_width=1,path_imgrec='mk_val.rec',shuffle=False,part_index=0,num_parts=1)

# train_iter.reset()
# for data in train_iter:
#     d = data.data[0]
#     l = np.unique(data.label[0])
#     print('shape:', d.shape)
#     print('label:', l)

# # take a look at some examples
# for i, batch in enumerate(val_iter):
#     d = batch.data[0]
#     l = batch.label[0]
#     data = d[0]
#     label = l[0]
#     print(data.shape)
#     data = mx.nd.transpose(data, (1,2,0))
#     plt.imshow(data.astype(np.uint8).asnumpy())
#     plt.show()
#     import ipdb; ipdb.set_trace()
#     if label == 0:
#         print("Forward")
#     else:
#         print("Backward")
#     if i == 20:
#         break


from mxnet.gluon.model_zoo import vision as models
# lets use a pretrained squeezenet, this a model known for being decently good accuracy at a low computational cost
squeezenet = models.squeezenet1_1(pretrained=True, prefix="direction_", ctx=ctx)

# create a new copy of squeezenet, this time though only have 2 output classes (hotdog or not hotdog)
dognet = models.squeezenet1_1(classes=2, prefix="direction_")
dognet.collect_params().initialize(ctx=ctx)

# use the the features chunk of squeezenet, only leave the output untouched
dognet.features = squeezenet.features

# in the trainer, specify that we only want to update the output chunk of dognet
trainer = gluon.Trainer(dognet.output.collect_params(), 'sgd', {'learning_rate': .01})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# given guess z and label y, compute the loss
def unbalanced_loss(loss_func, z, y):
    # there are 3 times more images of forward than going backward :(
    positive_class_weight = 3
    regular_loss = loss_func(z, y)
    # convienently y is either 1 (hotdog) or 0 (not hotdog) so scaling is pretty simple
    scaled_loss = regular_loss * (1 + y*positive_class_weight)/positive_class_weight
    return scaled_loss

# return metrics string representation
def metric_str(names, accs):
    return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])
metric = mx.metric.create(['acc', 'f1'])

from mxnet.image import color_normalize

def evaluate(net, data_iter, ctx):
    data_iter.reset()
    print('inside of evaluate function:')
    for i, batch in enumerate(data_iter):
        print('batch%d' %i)
        data = color_normalize(batch.data[0]/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for i, x in enumerate(data):
            print('x number %d' % i)
            outputs.append(net(x))
        metric.update(label, outputs)
    out = metric.get()
    metric.reset()
    return out

# now lets train dognet, this will look very similar to other training loops we've done
epochs = 10
best_f1 = 0
log_interval = 100

# val_names, val_accs = evaluate(dognet, val_iter, ctx)
# print('[Initial] validation: %s'%(metric_str(val_names, val_accs)))

for epoch in range(epochs):
    print('epoch #', epoch)
    tic = time.time()
    train_iter.reset()
    btic = time.time()
    for i, batch in enumerate(train_iter):
        print('batch #:', i)
        # the model zoo models expect normalized images
        data = color_normalize(batch.data[0]/255,
                               mean=mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1)),
                               std=mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1)))
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        Ls = []
        with autograd.record():
            for x, y in zip(data, label):
                z = dognet(x)
                # rescale the loss based on class to counter the imbalance problem                
                L = unbalanced_loss(loss, z, y)
                # store the loss and do backward after we have done forward
                # on all GPUs for better speed on multiple GPUs.
                Ls.append(L)
                outputs.append(z)
            for L in Ls:
                L.backward()
        trainer.step(batch.data[0].shape[0])
        metric.update(label, outputs)
        if log_interval and not (i+1)%log_interval:
            names, accs = metric.get()
            print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                           epoch, i, batch_size/(time.time()-btic), metric_str(names, accs)))
        btic = time.time()


    names, accs = metric.get()
    metric.reset()
    print('[Epoch %d] training: %s'%(epoch, metric_str(names, accs)))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    val_names, val_accs = evaluate(dognet, val_iter, ctx)
    print('[Epoch %d] validation: %s'%(epoch, metric_str(val_names, val_accs)))

    if val_accs[1] > best_f1:
        best_f1 = val_accs[1]
        print('Best validation f1 found. Checkpointing...')
        dognet.save_params('dog-%d.params'%(epoch))

# from skimage.color import rgba2rgb
# import skimage.io as io
# def classify_hotdog(net, url):
#     I = io.imread(url)
#     if I.shape[2] == 4:
#         I = rgba2rgb(I)
#     image = mx.nd.array(I).astype(np.uint8)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image.asnumpy())
#     image = mx.image.resize_short(image, 256)
#     image, _ = mx.image.center_crop(image, (224, 224))
#     plt.subplot(1, 2, 2)
#     plt.imshow(image.asnumpy())
#     image = mx.image.color_normalize(image.astype(np.float32)/255,
#                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
#                                      std=mx.nd.array([0.229, 0.224, 0.225]))
#     image = mx.nd.transpose(image.astype('float32'), (2,1,0))
#     image = mx.nd.expand_dims(image, axis=0)
#     out = mx.nd.SoftmaxActivation(net(image))
#     print('Probabilities are: '+str(out[0].asnumpy()))
#     result = np.argmax(out.asnumpy())
#     outstring = ['Not hotdog!', 'Hotdog!']
#     print(outstring[result])

# dognet.collect_params().reset_ctx(mx.cpu())
# classify_hotdog(dognet, "http://del.h-cdn.co/assets/17/25/980x490/landscape-1498074256-delish-blt-dogs-01.jpg")
# classify_hotdog(dognet, "https://i.ytimg.com/vi/SfLV8hD7zX4/maxresdefault.jpg")

