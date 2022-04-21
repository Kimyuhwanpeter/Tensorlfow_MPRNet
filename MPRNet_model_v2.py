# -*- coding:utf-8 -*-
from Downsampling import *
import tensorflow as tf

def conv(out_channels, kernel_size, bias=False, stride = 1):
    return tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)

##---------- Resizing Modules ----------    
class DownSample(tf.keras.layers.Layer):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = tf.keras.layers.Conv2D(filters=in_channels+s_factor, kernel_size=1, strides=1, use_bias=False)

    def call(self, x):
        x = tf.keras.layers.experimental.preprocessing.Resizing(x.shape[1] // 2, x.shape[2] // 2)(x)
        x = self.down(x)
        return x

class SkipUpSample(tf.keras.layers.Layer):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear"),
                                tf.keras.layers.Conv2D(in_channels, 1, strides=1, use_bias=False)])

    def call(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class UpSample(tf.keras.layers.Layer):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear"),
                                tf.keras.layers.Conv2D(in_channels, 1, strides=1, use_bias=False)])

    def call(self, x):
        x = self.up(x)
        return x

##########################################################################
## Channel Attention Layer
class CALayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(filters=channel // reduction, kernel_size=1, use_bias=bias),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=channel, kernel_size=1, use_bias=bias),
                tf.keras.layers.Activation(tf.nn.sigmoid)]
        )

    def call(self, x):
        y = self.avg_pool(x)
        y = tf.expand_dims(y, 1)
        y = tf.expand_dims(y, 1)
        y = self.conv_du(y)
        return x * y
##########################################################################

##########################################################################
## Channel Attention Block (CAB)
class CAB(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        #modules_body = []
        #modules_body.append(conv(n_feat, kernel_size, bias=bias))
        #modules_body.append(act)
        #modules_body.append(conv(n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        #self.body = tf.keras.Sequential([conv(n_feat, kernel_size, bias=bias), 
        #                                 act,
        #                                 conv(n_feat, kernel_size, bias=bias)])
        self.conv1 = conv(n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, kernel_size, bias=bias)
        self.act = act
        if self.act == "prelu":
            self.activa = tf.keras.layers.PReLU()
        if self.act == "relu" or self.act is None:
            self.activa = tf.keras.layers.ReLU()

    def call(self, x):
        res = self.conv1(x)
        #res = self.act(res)
        res = self.activa(res)
        res = self.conv2(res)
        res = self.CA(res)
        res += x
        return res

##########################################################################

##########################################################################
## Supervised Attention Module
class SAM(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, kernel_size, bias=bias)
        self.conv2 = conv(3, kernel_size, bias=bias)
        self.conv3 = conv(n_feat, kernel_size, bias=bias)

    def call(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = tf.nn.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################

## U-Net

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = tf.keras.Sequential(self.encoder_level1)
        self.encoder_level2 = tf.keras.Sequential(self.encoder_level2)  
        self.encoder_level3 = tf.keras.Sequential(self.encoder_level3)
        
        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = tf.keras.layers.Conv2D(n_feat, kernel_size=1, use_bias=bias)
            self.csff_enc2 = tf.keras.layers.Conv2D(n_feat+scale_unetfeats, kernel_size=1, use_bias=bias)
            self.csff_enc3 = tf.keras.layers.Conv2D(n_feat+(scale_unetfeats*2), kernel_size=1, use_bias=bias)

            self.csff_dec1 = tf.keras.layers.Conv2D(n_feat,kernel_size=1, use_bias=bias)
            self.csff_dec2 = tf.keras.layers.Conv2D(n_feat+scale_unetfeats,kernel_size=1, use_bias=bias)
            self.csff_dec3 = tf.keras.layers.Conv2D(n_feat+(scale_unetfeats*2), kernel_size=1, use_bias=bias)

    def call(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
        
        return [enc1, enc2, enc3]

class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = tf.keras.Sequential(self.decoder_level1)
        self.decoder_level2 = tf.keras.Sequential(self.decoder_level2)
        self.decoder_level3 = tf.keras.Sequential(self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def call(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]


##########################################################################
## Original Resolution Block (ORB)
class ORB(tf.keras.layers.Layer):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, kernel_size))
        self.body = tf.keras.Sequential(modules_body)

    def call(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(tf.keras.layers.Layer):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = tf.keras.Sequential([UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats)])
        self.up_dec2 = tf.keras.Sequential([UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats)])

        self.conv_enc1 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)
        self.conv_enc2 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)
        self.conv_enc3 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)

        self.conv_dec1 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)
        self.conv_dec2 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)
        self.conv_dec3 = tf.keras.layers.Conv2D(n_feat+scale_orsnetfeats, kernel_size=1, use_bias=bias)

    def call(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x
##########################################################################

class MPR_Net(tf.keras.Model):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(MPR_Net, self).__init__()

        act="prelu"
        self.shallow_feat1 = tf.keras.Sequential([conv(n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat2 = tf.keras.Sequential([conv(n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act)])
        self.shallow_feat3 = tf.keras.Sequential([conv(n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act)])

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail     = conv(out_c, kernel_size, bias=bias)

    def call(self, x3_img):
        # Original-resolution Image for Stage 3
        
        H = x3_img.shape[1]
        W = x3_img.shape[2]

        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches

        # Two Patches for Stage 2
        x2top_img  = x3_img[:,0:int(H/2),:,:]
        x2bot_img  = x3_img[:,int(H/2):H,:,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,0:int(W/2),:]
        x1rtop_img = x2top_img[:,:,int(W/2):W,:]
        x1lbot_img = x2bot_img[:,:,0:int(W/2),:]
        x1rbot_img = x2bot_img[:,:,int(W/2):W,:]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)
        x1rtop = self.shallow_feat1(x1rtop_img)
        x1lbot = self.shallow_feat1(x1lbot_img)
        x1rbot = self.shallow_feat1(x1rbot_img)
        
        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)
        
        ## Concat deep features
        feat1_top = [tf.concat((k,v), 2) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot = [tf.concat((k,v), 2) for k,v in zip(feat1_lbot,feat1_rbot)]
        
        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)

        ## Output image at Stage 1
        stage1_img = tf.concat([stage1_img_top, stage1_img_bot],1) 
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(tf.concat([x2top, x2top_samfeats], 3))
        x2bot_cat = self.concat12(tf.concat([x2bot, x2bot_samfeats], 3))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [tf.concat((k,v), 1) for k,v in zip(feat2_top,feat2_bot)]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)


        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x3_img)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(tf.concat([x3, x3_samfeats], 3))
        
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return [stage3_img+x3_img, stage2_img, stage1_img]
