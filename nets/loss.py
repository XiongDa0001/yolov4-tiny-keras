import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend import epsilon,clip,mean
from nets.ious import box_ciou



import numpy as np
from keras.backend import exp
from keras.layers import Softmax

import math


def DRloss(targets, logits):
    pos_lambda = 1
    margin = 0.5
    neg_lambda = 0.1 / math.log(3.5)
    L = 6.
    tau = 4.
    # 标注相关信息
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = np.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
    t = targets.unsqueeze(1)
    # 获得正负样本id
    pos_ind = (t == class_range)
    neg_ind = (t != class_range) * (t >= 0)
    # 概率p使用sigmoid求得
    pos_prob = logits[pos_ind].sigmoid()
    neg_prob = logits[neg_ind].sigmoid()
    # 对应于式(3.9)
    neg_q = Softmax(neg_prob/neg_lambda, dim=0)
    neg_dist = np.sum(neg_q * neg_prob)
    # 论文中提到，如果图像中没有正样本，则正样本的P使用1代替
    if pos_prob.numel() > 0:
        # 对应于式(3.9)
        pos_q = Softmax(-pos_prob/pos_lambda, dim=0)
        pos_dist = np.sum(pos_q * pos_prob)
        # 对应于式(3.12)
        loss = tau*np.log(1.+exp(L*(neg_dist - pos_dist+margin)))/L
    else:
        # 对应于式(3.12)
        loss = tau*np.log(1.+exp(L*(neg_dist - 1. + margin)))/L
    return loss


#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())

    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())

    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # 找出存在目标的先验框
        indices_for_object        = K.where(K.equal(anchor_state, 1))
        labels_for_object         = K.gather_nd(labels, indices_for_object)
        classification_for_object = K.gather_nd(classification, indices_for_object)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_object = K.ones_like(labels_for_object) * alpha
        alpha_factor_for_object = K.where(K.equal(labels_for_object, 1), alpha_factor_for_object, 1 - alpha_factor_for_object)
        focal_weight_for_object = K.where(K.equal(labels_for_object, 1), 1 - classification_for_object, classification_for_object)
        focal_weight_for_object = alpha_factor_for_object * focal_weight_for_object ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_object = focal_weight_for_object * K.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = K.where(K.equal(anchor_state, 0))
        labels_for_back         = K.gather_nd(labels, indices_for_back)
        classification_for_back = K.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        alpha_factor_for_back = K.ones_like(labels_for_back) * (1 - alpha)
        focal_weight_for_back = classification_for_back
        focal_weight_for_back = alpha_factor_for_back * focal_weight_for_back ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss_for_back = focal_weight_for_back * K.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = K.sum(cls_loss_for_object)
        cls_loss_for_back = K.sum(cls_loss_for_back)

        # 总的loss
        loss = (cls_loss_for_object + cls_loss_for_back)/normalizer

        return loss
    return _focal




def binary_focal_loss(true_label,probs ):
    gamma = 2
    alpha = 0.25
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    epsilon = 1.e-8
    # 得到y_true和y_pred
    y_true = tf.one_hot(true_label, 2)
    # probs = tf.nn.sigmoid(logits)
    y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
    # 得到调节因子weight和alpha
    ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
    p_t = y_true * y_pred \
          + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
    ## 然后通过p_t和gamma得到weight
    weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
    ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
    alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
    # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
    focal_loss = - alpha_t * weight * tf.log(p_t)
    return tf.reduce_mean(focal_loss)








#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    #---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    #---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


#---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
#---------------------------------------------------#
def box_iou(b1, b2):
    # 13,13,3,1,4
    # 计算左上角的坐标和右下角的坐标
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # 计算左上角和右下角的坐标
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 计算重合面积
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

#---------------------------------------------------#
#   loss值计算
#---------------------------------------------------#
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False, normalize=True):
    # 一共有两层
    num_layers = len(anchors)//3 

    #---------------------------------------------------------------------------------------------------#
    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    #   y_true是一个列表，包含两个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85)
    #   yolo_outputs是一个列表，包含两个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85)
    #---------------------------------------------------------------------------------------------------#
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    #-----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[81,82], [135,169], [344,319]
    #   26x26的特征层对应的anchor是[23,27], [37,58], [81,82]
    #-----------------------------------------------------------#
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    # 得到input_shpae为416,416 
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    loss = 0
    num_pos = 0
    #-----------------------------------------------------------#
    #   取出每一张图片
    #   m的值就是batch_size
    #-----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    #---------------------------------------------------------------------------------------------------#
    #   y_true是一个列表，包含两个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85)
    #   yolo_outputs是一个列表，包含两个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85)
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   以第一个特征层(m,13,13,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        #-----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]#[…, 4: 5]指的是该框的置信度,用1或0表示https://www.cnblogs.com/learningcaiji/p/14077315.html

        #-----------------------------------------------------------#
        #   取出其对应的种类(m,13,13,3,80)
        #-----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        #-----------------------------------------------------------#
        #   将yolo_outputs的特征层输出进行处理、获得四个返回值
        #   其中：
        #   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,85) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
        #-----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        
        #-----------------------------------------------------------#
        #   pred_box是解码后的预测的box的位置
        #   (m,13,13,3,4)
        #-----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        #-----------------------------------------------------------#
        #   找到负样本群组，第一步是创建一个数组，[]
        #-----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        
        #-----------------------------------------------------------#
        #   对每一张图片计算ignore_mask
        #-----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            #-----------------------------------------------------------#
            #   取出n个真实框：n,4
            #-----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            #-----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box    13,13,3,4 预测框的坐标
            #   true_box    n,4 真实框的坐标
            #   iou         13,13,3,n 预测框和真实框的iou
            #-----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            #-----------------------------------------------------------#
            #   best_iou    13,13,3 每个特征点与真实框的最大重合程度
            #-----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            #-----------------------------------------------------------#
            #   判断预测框和真实框的最大iou小于ignore_thresh
            #   则认为该预测框没有与之对应的真实框
            #   该操作的目的是：
            #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，所以忽略掉。
            #-----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        #-----------------------------------------------------------#
        #   在这个地方进行一个循环、循环是对每一张图片进行的
        #-----------------------------------------------------------#
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        #-----------------------------------------------------------#
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        #-----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        #-----------------------------------------------------------#
        #   真实框越大，比重越小，小框的比重更大。
        #-----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]  #预测值

        #-----------------------------------------------------------#
        #   计算Ciou loss
        #-----------------------------------------------------------#
        raw_true_box = y_true[l][...,0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)
        
        #------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        #------------------------------------------------------------------------------#

        #object_mask代表的是True
        #列表中的三个点代表前面所有的维数
        # print('object_ mask',object_mask.shape)
        # print('raw',raw_pred[...,4:5].shape)



        #confidence_loss，实际存在的框，预测结果中置信度的值与1对比；实际不存在的框，预测结果中置信度的值与0对比，该部分要去除被忽略的不包含目标的框。
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5],from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5],from_logits=True) * ignore_mask
        # confidence_loss = binary_focal_loss(object_mask,raw_pred[...,4:5])
        # print('confidence_loss',confidence_loss)
        #置信度loss
        '''正样本有坐标，置信度和类别损失函数，而负样本只有置信度损失函数'''

        '''
                alpha = 0.25
        gamma = 2
        alpha_factor = K.ones_like(object_mask) * alpha
        alpha_factor = tf.where(K.equal(object_mask, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(object_mask, 1), 1 - raw_pred[..., 4:5], raw_pred[..., 4:5])
        focal_weight = alpha_factor * focal_weight ** gamma
        confidence_loss = focal_weight * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],from_logits=True)
        '''


        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:],from_logits=True) #分类loss

        location_loss = K.sum(ciou_loss) #回归框
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        #-----------------------------------------------------------#
        #   计算正样本数量
        #-----------------------------------------------------------#
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += location_loss + confidence_loss + class_loss
        # if print_loss:
        #   loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
        
    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / mf
    return loss
