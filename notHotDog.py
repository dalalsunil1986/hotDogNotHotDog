#import Dependencies
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import time
# import matplotlib.pyplot as plt
import os
import logging
logging.basicConfig(level=logging.INFO)
# from mxnet.test_utils import download
from mxnet.gluon.model_zoo import vision as models
from mxnet.image import color_normalize

batch_size = 64
ctx = [mx.cpu()]

class MKDirectionDetect(object):
    def __init__(self):
        # lets use a pretrained squeezenet, this a model known for being decently good accuracy at a low computational cost
        squeezenet = models.squeezenet1_1(pretrained=True, prefix="direction_", ctx=ctx)

        # create a new copy of squeezenet, this time though only have 2 output classes (hotdog or not hotdog)
        self.dirNet = models.squeezenet1_1(classes=2, prefix="direction_")
        self.dirNet.collect_params().initialize(ctx=ctx)

        # use the the features chunk of squeezenet, only leave the output untouched
        self.dirNet.features = squeezenet.features

        # in the trainer, specify that we only want to update the output chunk of self.dirNet
        self.trainer = gluon.Trainer(self.dirNet.output.collect_params(), 'sgd', {'learning_rate': .01})
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

        self.metric = mx.metric.create(['acc', 'f1'])

    # given guess z and label y, compute the loss
    def unbalanced_loss(self, loss_func, z, y):
        # there are 3 times more images of forward than going backward :(
        positive_class_weight = 3
        regular_loss = loss_func(z, y)
        # convienently y is either 1 (hotdog) or 0 (not hotdog) so scaling is pretty simple
        scaled_loss = regular_loss * (1 + y*positive_class_weight)/positive_class_weight
        return scaled_loss

    # return metrics string representation
    def metric_str(self, names, accs):
        return ', '.join(['%s=%f'%(name, acc) for name, acc in zip(names, accs)])

    def evaluate(self, net, data_iter, ctx):
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
            self.metric.update(label, outputs)
        out = self.metric.get()
        self.metric.reset()
        return out

    def train(self):
        # now lets train self.dirNet, this will look very similar to other training loops we've done
        # Training and validation iterators
        train_iter = mx.io.ImageRecordIter(batch_size=batch_size, data_shape=(3,270,270), label_width=1, path_imgrec='mk_train.rec', shuffle=True)
        val_iter = mx.io.ImageRecordIter(batch_size=batch_size, data_shape=(3,270,270), label_width=1, path_imgrec='mk_val.rec', shuffle=False)

        epochs = 10
        best_f1 = 0
        log_interval = 100

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
                        z = self.dirNet(x)
                        # rescale the loss based on class to counter the imbalance problem                
                        L = self.unbalanced_loss(self.loss, z, y)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                        Ls.append(L)
                        outputs.append(z)
                    for L in Ls:
                        L.backward()
                self.trainer.step(batch.data[0].shape[0])
                self.metric.update(label, outputs)
                if log_interval and not (i+1)%log_interval:
                    names, accs = self.metric.get()
                    print('[Epoch %d Batch %d] speed: %f samples/s, training: %s'%(
                                epoch, i, batch_size/(time.time()-btic), self.metric_str(names, accs)))
                btic = time.time()

            names, accs = self.metric.get()
            self.metric.reset()
            print('[Epoch %d] training: %s'%(epoch, self.metric_str(names, accs)))
            print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
            val_names, val_accs = self.evaluate(self.dirNet, val_iter, ctx)
            print('[Epoch %d] validation: %s'%(epoch, self.metric_str(val_names, val_accs)))

            if val_accs[1] > best_f1:
                best_f1 = val_accs[1]
                print('Best validation f1 found. Checkpointing...')
                self.dirNet.save_params('dog-%d.params'%(epoch))

    def classify_hotdog(self, url, params=None):
        # self.dirNet.collect_params().reset_ctx(mx.cpu())
        if(params is not None):
            print('params is not None good job')
            self.dirNet.load_params(params, ctx=ctx)
        else:
            # self.dirNet.collect_params().initialize(mx.init.Xavier(), ctx=self.ctx)
            self.dirNet.collect_params().initialize(ctx=ctx)
        
        from skimage.color import rgba2rgb
        import skimage.io as io
        import cv2
        I = io.imread(url)
        if I.shape[2] == 4:
            I = rgba2rgb(I)
        image = mx.nd.array(I).astype(np.uint8)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image.asnumpy())
        
        
        # image = mx.image.resize_short(image, 256)
        # image, _ = mx.image.center_crop(image, (224, 224))
        # plt.subplot(1, 2, 2)
        # plt.imshow(image.asnumpy())
        
        # imageToDisplay = cv2.imread(url)
        # cv2.imshow("image", imageToDisplay)
        # cv2.waitKey(0) #press 0 to exit

        image = mx.image.color_normalize(image.astype(np.float32)/255,
                                            mean=mx.nd.array([0.485, 0.456, 0.406]),
                                            std=mx.nd.array([0.229, 0.224, 0.225]))
        image = mx.nd.transpose(image.astype('float32'), (2,1,0))
        image = mx.nd.expand_dims(image, axis=0)
        out = mx.nd.SoftmaxActivation(self.dirNet(image))
        print('Probabilities are: '+str(out[0].asnumpy()))
        result = np.argmax(out.asnumpy())
        outstring = ['Right Direction!', 'Wrong Direction!']
        print(outstring[result])

def main():
    print('Starting Extraction')
    model_params = './params/dog-9.params'
    directionModel = MKDirectionDetect()
    # directionModel.train()
    directionModel.classify_hotdog("videoFrames/video0/frame22.jpg", model_params)
    directionModel.classify_hotdog("videoFrames/video1/frame22.jpg", model_params)
    directionModel.classify_hotdog("videoFrames/video0/frame24.jpg", model_params)
    directionModel.classify_hotdog("videoFrames/video1/frame24.jpg", model_params)

if __name__ == '__main__':
    main()
