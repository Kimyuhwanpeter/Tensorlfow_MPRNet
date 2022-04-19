# -*- coding:utf-8 -*-
import tensorflow as tf

def Channel_attention_layer(input, filters, reduction=4, use_bias=False):


    y = tf.keras.layers.GlobalAveragePooling2D()(input)
    y = tf.expand_dims(y, 1)
    y = tf.expand_dims(y, 1)
    y = tf.keras.layers.Conv2D(filters=filters // reduction, 
                               kernel_size=1, 
                               use_bias=use_bias)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Conv2D(filters=filters, 
                               kernel_size=1, 
                               use_bias=use_bias)(y)
    y = tf.nn.sigmoid(y)

    return y * input

def CAB(input, filters, kernel_size=3, reduction=4, use_bias=False):

    h = tf.keras.layers.Conv2D(filters=filters, 
                               kernel_size=kernel_size,
                               padding="same", use_bias=use_bias)(input)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters, 
                               kernel_size=kernel_size,
                               padding="same", use_bias=use_bias)(h)
    h = Channel_attention_layer(input=h, filters=filters, reduction=reduction, use_bias=use_bias)

    h += input

    return h

def SAM(x, x_img, filters, kernel_size=1, use_bias=False):

    x1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                use_bias=use_bias)(x)
    img = tf.keras.layers.Conv2D(filters=3, kernel_size=kernel_size,
                                 use_bias=use_bias)(x) + x_img
    x2 = tf.nn.sigmoid(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                              use_bias=use_bias)(img))
    x1 = x1 * x2
    x1 = x1 + x

    return x1, img

def ORB(input, filters, kernel_size=3, reduction=4, use_bias=False, num_cab=8):

    res = input
    for _ in range(num_cab):
        res = CAB(res, filters, kernel_size, reduction, False)
    res = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                 padding="same", use_bias=False)(res)
    
    res += input

    return res

def Encoder(input, filters=96, kernel_size=3, scale_unetfeats=48, reduction=4,
            encoder_outs=None, decoder_outs=None):

    encoder_level1 = CAB(input, filters, kernel_size, reduction, False)
    encoder_level1 = CAB(encoder_level1, filters, kernel_size, reduction, False)
    enc1 = encoder_level1
    if (encoder_outs is not None) and (decoder_outs is not None):
        enc1 = enc1 + tf.keras.layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(encoder_outs[0]) \
            + tf.keras.layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(decoder_outs[0])

    x = tf.image.resize(enc1, (enc1.shape[1] // 2, enc1.shape[2] // 2))
    x = tf.keras.layers.Conv2D(filters=filters+scale_unetfeats, kernel_size=1, use_bias=False)(x)

    enc2 = CAB(x, filters+scale_unetfeats, kernel_size, reduction, False)
    enc2 = CAB(enc2, filters+scale_unetfeats, kernel_size, reduction, False)

    if (encoder_outs is not None) and (decoder_outs is not None):
        enc2 = enc2 + tf.keras.layers.Conv2D(filters=filters+scale_unetfeats, kernel_size=1, use_bias=False)(encoder_outs[1]) \
            + tf.keras.layers.Conv2D(filters=filters+scale_unetfeats, kernel_size=1, use_bias=False)(decoder_outs[1])

    x = tf.image.resize(enc2, (enc2.shape[1] // 2, enc2.shape[2] // 2))
    x = tf.keras.layers.Conv2D(filters=filters+(scale_unetfeats*2), kernel_size=1, use_bias=False)(x)

    enc3 = CAB(x, filters+(scale_unetfeats*2), kernel_size, reduction, False)
    enc3 = CAB(enc3, filters+(scale_unetfeats*2), kernel_size, reduction, False)

    if (encoder_outs is not None) and (decoder_outs is not None):
        enc3 = enc3 + tf.keras.layers.Conv2D(filters=filters+(scale_unetfeats*2), kernel_size=1, use_bias=False)(encoder_outs[2]) \
            + tf.keras.layers.Conv2D(filters=filters+(scale_unetfeats*2), kernel_size=1, use_bias=False)(decoder_outs[2])

    return enc1, enc2, enc3

def Decoder(input, filters=96, kernel_size=3, scale_unetfeats=48, reduction=4):

    enc1, enc2, enc3 = input

    dec3 = CAB(enc3, filters+(scale_unetfeats*2), kernel_size, reduction, False)
    dec3 = CAB(dec3, filters+(scale_unetfeats*2), kernel_size, reduction, False)

    x = tf.image.resize(dec3, (dec3.shape[1] * 2, dec3.shape[2] * 2))
    x = tf.keras.layers.Conv2D(filters=filters+scale_unetfeats, kernel_size=1,
                               padding="same", use_bias=False)(x)
    skip_attn2 = CAB(enc2, filters+scale_unetfeats, kernel_size, reduction, False)
    x = skip_attn2 + x

    dec2 = CAB(x, filters+scale_unetfeats, kernel_size, reduction, False)
    dec2 = CAB(dec2, filters+scale_unetfeats, kernel_size, reduction, False)

    x = tf.image.resize(dec2, (dec2.shape[1] * 2, dec2.shape[2] * 2))
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                               padding="same", use_bias=False)(x)

    skip_attn1 = CAB(enc1, filters, kernel_size, reduction, False)
    x = skip_attn1 + x

    dec1 = CAB(x, filters, kernel_size, reduction, False)
    dec1 = CAB(dec1, filters, kernel_size, reduction, False)

    return [dec1, dec2, dec3]

def ORSNet(x, encoder_outs, decoder_outs, filters, kernel_size, scale_orsnetfeats, reduction, scale_unetfeats, num_cab):

    x = ORB(input=x,filters=filters+scale_orsnetfeats, kernel_size=kernel_size, reduction=reduction, use_bias=False, num_cab=num_cab)
    x = x + tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(encoder_outs[0]) + \
        tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(decoder_outs[0])

    x = ORB(input=x,filters=filters+scale_orsnetfeats, kernel_size=kernel_size, reduction=reduction, use_bias=False, num_cab=num_cab)
    encoder_outs1 = tf.image.resize(encoder_outs[1], [encoder_outs[1].shape[1] * 2, encoder_outs[1].shape[2] * 2])
    encoder_outs1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                           use_bias=False)(encoder_outs1)
    decoder_outs1 = tf.image.resize(decoder_outs[1], [decoder_outs[1].shape[1] * 2, decoder_outs[1].shape[2] * 2])
    decoder_outs1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                           use_bias=False)(decoder_outs1)
    x = x + tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(encoder_outs1) + \
        tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(decoder_outs1)

    x = ORB(input=x,filters=filters+scale_orsnetfeats, kernel_size=kernel_size, reduction=reduction, use_bias=False, num_cab=num_cab)
    encoder_outs2 = tf.image.resize(encoder_outs[2], [encoder_outs[2].shape[1] * 2, encoder_outs[2].shape[2] * 2])
    encoder_outs2 = tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats, kernel_size=1,
                                           use_bias=False)(encoder_outs2)
    encoder_outs2 = tf.image.resize(encoder_outs2, [encoder_outs2.shape[1] * 2, encoder_outs2.shape[2] * 2])
    encoder_outs2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                           use_bias=False)(encoder_outs2)

    decoder_outs2 = tf.image.resize(decoder_outs[2], [decoder_outs[2].shape[1] * 2, decoder_outs[2].shape[2] * 2])
    decoder_outs2 = tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats, kernel_size=1,
                                           use_bias=False)(decoder_outs2)
    decoder_outs2 = tf.image.resize(decoder_outs2, [decoder_outs2.shape[1] * 2, decoder_outs2.shape[2] * 2])
    decoder_outs2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=1,
                                           use_bias=False)(decoder_outs2)

    x = x + tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(encoder_outs2) + \
        tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats,
                                   kernel_size=1,
                                   use_bias=False)(decoder_outs2)
    return x

def MPRNet(inputs_shape=(512, 512, 3), filters=80, reduction=4, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8):

    h = inputs = tf.keras.Input(inputs_shape)

    h_2_top = h[:, 0:int(inputs_shape[0]//2), :, :]
    h_2_bot = h[:, int(inputs_shape[0]//2):inputs_shape[0], :, :]

    h_1_ltop = h_2_top[:, :, 0:int(inputs_shape[1] // 2), :]
    h_1_rtop = h_2_top[:, :, int(inputs_shape[1] // 2):inputs_shape[1], :]
    h_1_lbot = h_2_bot[:, :, 0:int(inputs_shape[1] // 2), :]
    h_1_rbot = h_2_bot[:, :, int(inputs_shape[1] // 2):inputs_shape[1], :]

    h_1_ltop = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_1_ltop)
    h_1_ltop = CAB(h_1_ltop, filters, 3, reduction, False)
    h_1_ltop_enc1, h_1_ltop_enc2, h_1_ltop_enc3 = Encoder(h_1_ltop, filters)
    feat1_ltop = [h_1_ltop_enc1, h_1_ltop_enc2, h_1_ltop_enc3]

    h_1_rtop = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_1_rtop)
    h_1_rtop = CAB(h_1_rtop, filters, 3, reduction, False)
    h_1_rtop_enc1, h_1_rtop_enc2, h_1_rtop_enc3 = Encoder(h_1_rtop, filters)
    feat1_rtop = [h_1_rtop_enc1, h_1_rtop_enc2, h_1_rtop_enc3]

    h_1_lbot = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_1_lbot)
    h_1_lbot = CAB(h_1_lbot, filters, 3, reduction, False)
    h_1_lbot_enc1, h_1_lbot_enc2, h_1_lbot_enc3 = Encoder(h_1_lbot, filters)
    feat1_lbot = [h_1_lbot_enc1, h_1_lbot_enc2, h_1_lbot_enc3]

    h_1_rbot = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_1_rbot)
    h_1_rbot = CAB(h_1_rbot, filters, 3, reduction, False)
    h_1_rbot_enc1, h_1_rbot_enc2, h_1_rbot_enc3 = Encoder(h_1_rbot, filters)
    feat1_rbot = [h_1_rbot_enc1, h_1_rbot_enc2, h_1_rbot_enc3]

    feat1_top = [tf.concat([k, v], 2) for k,v in zip(feat1_ltop, feat1_rtop)]
    feat1_bot = [tf.concat([k, v], 2) for k,v in zip(feat1_lbot, feat1_rbot)]

    res1_top = Decoder(feat1_top, filters)
    res1_bot = Decoder(feat1_bot, filters)

    h2top_samfeats, stage1_img_top = SAM(res1_top[0], h_2_top, filters)
    h2bot_samfeats, stage1_img_bot = SAM(res1_bot[0], h_2_bot, filters)

    stage1_img = tf.concat([stage1_img_top, stage1_img_bot], 1)

    ##########################################################################################

    h_2_top_ = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_2_top)
    h_2_top_ = CAB(h_2_top_, filters, 3, reduction, False)

    h_2_bot_ = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h_2_bot)
    h_2_bot_ = CAB(h_2_bot_, filters, 3, reduction, False)

    h2_top_cat = tf.concat([h_2_top_, h2top_samfeats], 3)
    h2_top_cat = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                        padding="same", use_bias=False)(h2_top_cat)
    h2_top_cat_enc1, h2_top_cat_enc2, h2_top_cat_enc3 = Encoder(h2_top_cat, filters,
                                                                encoder_outs=feat1_top,
                                                                decoder_outs=res1_top)
    feat2_top = [h2_top_cat_enc1, h2_top_cat_enc2, h2_top_cat_enc3]

    h2_bot_cat = tf.concat([h_2_bot_, h2bot_samfeats], 3)
    h2_bot_cat = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                        padding="same", use_bias=False)(h2_bot_cat)
    h2_bot_cat_enc1, h2_bot_cat_enc2, h2_bot_cat_enc3 = Encoder(h2_bot_cat, filters,
                                                                encoder_outs=feat1_bot,
                                                                decoder_outs=res1_bot)
    feat2_bot = [h2_bot_cat_enc1, h2_bot_cat_enc2, h2_bot_cat_enc3]

    feat2 = [tf.concat((k,v), 1) for k, v in zip(feat2_top, feat2_bot)]

    res2 = Decoder(feat2, filters)

    h3_samfeats, stage2_img = SAM(res2[0], h, filters)

    ##########################################################################################

    h_3 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                               padding="same", use_bias=False)(h)
    h_3 = CAB(h_3, filters, 3, reduction, False)
    h_3_cat = tf.concat([h_3, h3_samfeats], 3)
    h_3_cat = tf.keras.layers.Conv2D(filters=filters+scale_orsnetfeats, kernel_size=3,
                                        padding="same", use_bias=False)(h_3_cat)
    
    h_3_cat = ORSNet(h_3_cat, feat2, res2, filters, 3, scale_orsnetfeats, reduction, scale_unetfeats, num_cab)

    stage3_img = tf.keras.layers.Conv2D(filters=3, kernel_size=3,
                               padding="same", use_bias=False)(h_3_cat)

    return tf.keras.Model(inputs=inputs, outputs=[stage3_img+h, stage2_img, stage1_img])
