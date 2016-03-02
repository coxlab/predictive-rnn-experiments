

# Here are the two loss functions, just put in objectives.py
def GAN_generator_loss2(y_true, y_pred):
    #y_true should be a two column vector with second column all ones
    return -1.*T.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)


def GAN_discriminator_loss(y_true, y_pred):
  return -1.*T.log( epsilon + (y_true*y_pred).sum(axis=-1) ).mean(axis=-1)



# Then when you're training, just do something like this.
y_D = np.zeros((2*P['batch_size'],2), int)
# first samples will be real
y_D[:P['batch_size'],1] = 1
# the next will be samples from the generator
y_D[P['batch_size']:,0] = 1

y_G = np.zeros((P['batch_size'],2), int)
y_G[:,1] = 1


# Once you have the stuff in keras, you can do stuff like this
D_model.add_node(Dense(128, 2), name='fc1_fus' , input='fc0_fus_relu')
# This will make sure the weights are shared and they won't update when training the generator
G_model.add_node(Dense(128, 2, shared_weights_layer=D_model.nodes['fc1_fus'], params_fixed=True), name='fc1_fus' , input='fc0_fus_relu')
G_model.add_node(Activation('softmax'), name='D_softmax', input='fc1_fus')
G_model.add_output(name='output', input='D_softmax')

# Also I forgot to mention, but it will prob be really helpful to do a weighted GAN and MSE loss
# I implemented this by hacking the keras code.  I think there was actually a way to do it already, but I felt this was easier
model.compile(optimizer='sgd', loss={'output': 'GAN_generator_loss2', 'pixel_output': 'mse'}, obj_weights={'output': 0.01, 'pixel_output': 1.0})
