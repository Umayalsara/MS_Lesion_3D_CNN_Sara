import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    shortcut = x
    x = layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding, strides=1, activation='relu')(x)
    shortcut = layers.Conv3D(filters, kernel_size=1, padding=padding, strides=strides)(shortcut)
    x = layers.add([x, shortcut])
    return x

def attention_block(x, g, inter_channel):
    theta_x = layers.Conv3D(inter_channel, kernel_size=2, strides=2, padding='same')(x)
    phi_g = layers.Conv3D(inter_channel, kernel_size=1, padding='same')(g)
    
    print(f"Shape of theta_x: {theta_x.shape}")
    print(f"Shape of phi_g before upsampling: {phi_g.shape}")
    
    # Calculate the upsampling size
    scale_factors = (
        max(int(theta_x.shape[1] / phi_g.shape[1]), 1),
        max(int(theta_x.shape[2] / phi_g.shape[2]), 1),
        max(int(theta_x.shape[3] / phi_g.shape[3]), 1),
    )
    
    # Adjust the shape of phi_g to match theta_x
    if scale_factors != (1, 1, 1):
        phi_g = layers.UpSampling3D(size=scale_factors)(phi_g)
    
    print(f"Shape of phi_g after upsampling: {phi_g.shape}")
    
    concat_xg = layers.add([theta_x, phi_g])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv3D(1, kernel_size=1, padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    upsample_psi = layers.UpSampling3D(size=(2, 2, 2))(sigmoid_xg)
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv3D(inter_channel, kernel_size=1, padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    e1 = residual_block(inputs, 32)
    e2 = layers.MaxPooling3D((2, 2, 2), padding='same')(e1)
    e2 = residual_block(e2, 64)
    e3 = layers.MaxPooling3D((2, 2, 2), padding='same')(e2)
    e3 = residual_block(e3, 128)
    e4 = layers.MaxPooling3D((2, 2, 2), padding='same')(e3)
    e4 = residual_block(e4, 256)
    
    # Decoder with attention
    d1 = layers.UpSampling3D((2, 2, 2))(e4)
    d1 = attention_block(d1, e3, 128)
    d1 = layers.concatenate([d1, e3])
    d1 = residual_block(d1, 128)
    
    d2 = layers.UpSampling3D((2, 2, 2))(d1)
    d2 = attention_block(d2, e2, 64)
    d2 = layers.concatenate([d2, e2])
    d2 = residual_block(d2, 64)
    
    d3 = layers.UpSampling3D((2, 2, 2))(d2)
    d3 = attention_block(d3, e1, 32)
    d3 = layers.concatenate([d3, e1])
    d3 = residual_block(d3, 32)
    
    outputs = layers.Conv3D(1, kernel_size=1, activation='sigmoid')(d3)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
    return model

if __name__ == "__main__":
    input_shape = (32, 32, 32, 1)
    model = build_model(input_shape)
    model.summary()

    # Save the model architecture to a file in the native Keras format
    model.save('/nesi/project/uoa04272/software/tensorflow-2.17.0/MS_DETECTION_3D_CNN/model_architecture.keras')
