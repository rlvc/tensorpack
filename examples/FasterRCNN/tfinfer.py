from genericpath import isfile
import time
import cv2
from dataset import DatasetRegistry, register_coco, register_balloon
import numpy as np
import os
os.environ['KMP_WARNINGS'] = '0'
os.environ['ENFLAME_LOG_LEVEL'] = 'FATAL'
os.environ['SDK_LOG_LEVEL'] = '3'
import tensorflow as tf
from config import config as cfg
from eval import _paste_mask, DetectionResult
from common import CustomResize, clip_boxes
import tensorpack.utils.viz as tpviz
from tensorpack.utils import logger
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)
from config import finalize_configs
# import argparse
# import TopsInference
model_path = "/tmp/maskrcnn_1600.pb"
if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    finalize_configs(is_training=False)
    img = cv2.imread("./sq.jpg", cv2.IMREAD_COLOR)
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img).astype(np.float32)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    # resized_img1 = np.load("/tmp/input.npy").astype(np.float32)
        
    with tf.device("/device:CPU:0"):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                sess.run(tf.compat.v1.global_variables_initializer())
                aaa = []
                for i in range(10):
                    print("CURRENT {} ROUND".format(i + 1))
                    
                    # standard input
                    input = sess.graph.get_tensor_by_name('image:0')
                    boxes = sess.graph.get_tensor_by_name('output/boxes:0')
                    scores = sess.graph.get_tensor_by_name('output/scores:0')
                    labels = sess.graph.get_tensor_by_name('output/labels:0')
                    masks = sess.graph.get_tensor_by_name('output/masks:0')
                    
                    time_total_begin = time.time()
                    foutputs = sess.run(
                                        [
                                            boxes, scores, labels, masks,
                                        ],
                                        feed_dict={
                                            input: resized_img
                                        })

                    time_total_end = time.time()
                    print("[TIME] TOTAL time-consuming: {:.3f}ms"
                    .format((time_total_end - time_total_begin) * 1000))
                    aaa.append((time_total_end - time_total_begin) * 1000)

                print(np.mean(aaa[1:]))
                boxes = foutputs[0]
                scores = foutputs[1]
                labels = foutputs[2]
                masks = foutputs[3]

                # Some slow numpy postprocessing:
                boxes = boxes / scale
                # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
                boxes = clip_boxes(boxes, orig_shape)
                full_masks = [_paste_mask(box, mask, orig_shape)
                            for box, mask in zip(boxes, masks)]
                masks = full_masks
                results = [DetectionResult(*args) for args in zip(boxes, scores, labels.tolist(), masks)]
                final = draw_final_outputs_blackwhite(img, results)
                viz = np.concatenate((img, final), axis=1)
                cv2.imwrite("output.png", viz)
                # logger.info("Inference output for {} written to output.png".format(input_file))
                tpviz.interactive_imshow(viz)
