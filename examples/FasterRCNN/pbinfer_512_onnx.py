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
import TopsInference
model_path = "/tmp/maskrcnn_512.pb"
if __name__ == '__main__':
    register_coco(cfg.DATA.BASEDIR)
    finalize_configs(is_training=False)
    img = cv2.imread("./sq.jpg", cv2.IMREAD_COLOR)
    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img).astype(np.float32)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    # resized_img1 = np.load("/tmp/input.npy").astype(np.float32)
    with TopsInference.device(0, 0):
        os.environ['DTU_UMD_FLAGS'] = 'ib_pool_size=209715200'
        
        ## BACKBONE
        if(os.path.isfile("./engines512/backbone_onnx_32.exec")):
             engine_backbone = TopsInference.load("./engines512/backbone_onnx_32.exec")
        else:
            tf_parser_backbone = TopsInference.create_parser(TopsInference.ONNX_MODEL)
            network_backbone = tf_parser_backbone.read("/tmp/backbone_sim.onnx")
            optimizer = TopsInference.create_optimizer()
            # optimizer.set_build_flag(TopsInference.KFP16_MIX)
            engine_backbone = optimizer.build(network_backbone)
            engine_backbone.save_executable("./engines512/backbone_onnx_32.exec")
       
        #FC cascade_rcnn_stage1
        if(os.path.isfile("./engines512/fc_1.exec")):
             engine_fc_1 = TopsInference.load("./engines512/fc_1.exec")
        else:
            tf_parser_fc_1 = TopsInference.create_parser(TopsInference.TF_MODEL)
            tf_parser_fc_1.set_input_names(['cascade_rcnn_stage1/multilevel_roi_align/output'])
            tf_parser_fc_1.set_input_shapes([[1000, 256, 7, 7]])
            tf_parser_fc_1.set_output_names(['cascade_rcnn_stage1/head/fc7/output'])
            network_fc_1 = tf_parser_fc_1.read(model_path)
            optimizer = TopsInference.create_optimizer()
            engine_fc_1 = optimizer.build(network_fc_1)
            engine_fc_1.save_executable("./engines512/fc_1.exec")

        #FC cascade_rcnn_stage2
        if(os.path.isfile("./engines512/fc_2.exec")):
             engine_fc_2 = TopsInference.load("./engines512/fc_2.exec")
        else:
            tf_parser_fc_2 = TopsInference.create_parser(TopsInference.TF_MODEL)
            tf_parser_fc_2.set_input_names(['cascade_rcnn_stage2/multilevel_roi_align/output'])
            tf_parser_fc_2.set_input_shapes([[1000, 256, 7, 7]])
            tf_parser_fc_2.set_output_names(['cascade_rcnn_stage2/head/fc7/output'])
            network_fc_2 = tf_parser_fc_2.read(model_path)
            optimizer = TopsInference.create_optimizer()
            engine_fc_2 = optimizer.build(network_fc_2)
            engine_fc_2.save_executable("./engines512/fc_2.exec")

        #FC cascade_rcnn_stage3
        if(os.path.isfile("./engines512/fc_3.exec")):
             engine_fc_3 = TopsInference.load("./engines512/fc_3.exec")
        else:
            tf_parser_fc_3 = TopsInference.create_parser(TopsInference.TF_MODEL)
            tf_parser_fc_3.set_input_names(['cascade_rcnn_stage3/multilevel_roi_align/output'])
            tf_parser_fc_3.set_input_shapes([[1000, 256, 7, 7]])
            tf_parser_fc_3.set_output_names(['cascade_rcnn_stage3/head/fc7/output'])
            network_fc_3 = tf_parser_fc_3.read(model_path)
            optimizer = TopsInference.create_optimizer()
            engine_fc_3 = optimizer.build(network_fc_3)
            engine_fc_3.save_executable("./engines512/fc_0.exec")

        #FC maskrcnn
        if(os.path.isfile("./engines512/fc_maskrcnn.exec")):
             engine_fc_maskrcnn = TopsInference.load("./engines512/fc_maskrcnn.exec")
        else:
            tf_parser_fc_maskrcnn = TopsInference.create_parser(TopsInference.TF_MODEL)
            tf_parser_fc_maskrcnn.set_input_names(['multilevel_roi_align/output'])
            tf_parser_fc_maskrcnn.set_input_shapes([[10, 256, 14, 14]])
            tf_parser_fc_maskrcnn.set_output_names(['maskrcnn/fcn3/output'])
            network_fc_maskrcnn = tf_parser_fc_maskrcnn.read(model_path)
            optimizer = TopsInference.create_optimizer()
            engine_fc_maskrcnn = optimizer.build(network_fc_maskrcnn)
            engine_fc_maskrcnn.save_executable("./engines512/fc_maskrcnn.exec")
        
        with tf.device("/device:CPU:0"):
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                with tf.gfile.FastGFile(model_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    sess.run(tf.compat.v1.global_variables_initializer())
                    backbone_time = []
                    total_time = []
                    for i in range(10):
                        print("="*88)
                        print("CURRENT {} ROUND".format(i + 1))
                        print("="*88)
                        time_total_begin = time.time()
                        time_begin = time.time()
                        # # standard input
                        # input = sess.graph.get_tensor_by_name('image:0')

                        # standard output
                        boxes = sess.graph.get_tensor_by_name('output/boxes:0')
                        scores = sess.graph.get_tensor_by_name('output/scores:0')
                        labels = sess.graph.get_tensor_by_name('output/labels:0')
                        masks = sess.graph.get_tensor_by_name('output/masks:0')

                        # backbone output
                        input_fpn_out_2 = sess.graph.get_tensor_by_name(
                            'fpn/posthoc_3x3_p2/output:0')
                        input_fpn_out_3 = sess.graph.get_tensor_by_name(
                            'fpn/posthoc_3x3_p3/output:0')
                        input_fpn_out_4 = sess.graph.get_tensor_by_name(
                            'fpn/posthoc_3x3_p4/output:0')
                        input_fpn_out_5 = sess.graph.get_tensor_by_name(
                            'fpn/posthoc_3x3_p5/output:0') 
                        
                        Lvl6_Reshape_1 = sess.graph.get_tensor_by_name(
                            'generate_fpn_proposals/Lvl6/Reshape_1:0')
                        Lvl5_Reshape_1 = sess.graph.get_tensor_by_name(
                            'generate_fpn_proposals/Lvl5/Reshape_1:0')
                        Lvl4_Reshape_1 = sess.graph.get_tensor_by_name(
                            'generate_fpn_proposals/Lvl4/Reshape_1:0')
                        Lvl3_Reshape_1 = sess.graph.get_tensor_by_name(
                            'generate_fpn_proposals/Lvl3/Reshape_1:0')
                        Lvl2_Reshape_1 = sess.graph.get_tensor_by_name(
                            'generate_fpn_proposals/Lvl2/Reshape_1:0')
                        rpn_trans = sess.graph.get_tensor_by_name('rpn/transpose_1:0')
                        rpn1_trans = sess.graph.get_tensor_by_name('rpn_1/transpose_1:0')
                        rpn2_trans = sess.graph.get_tensor_by_name('rpn_2/transpose_1:0')
                        rpn3_trans = sess.graph.get_tensor_by_name('rpn_3/transpose_1:0')
                        rpn4_trans = sess.graph.get_tensor_by_name('rpn_4/transpose_1:0')
                        # crop_resize_1 input

                        # crop_resize_1 output
                        
                        #fc_1
                        fpn_box = sess.graph.get_tensor_by_name('generate_fpn_proposals/boxes:0')
                        stage_1_box = sess.graph.get_tensor_by_name('cascade_rcnn_stage1/output_boxes:0')
                        stage_2_box = sess.graph.get_tensor_by_name('cascade_rcnn_stage2/output_boxes:0')
                        fc_1_in = sess.graph.get_tensor_by_name('cascade_rcnn_stage1/multilevel_roi_align/output:0')
                        fc_1_out = sess.graph.get_tensor_by_name('cascade_rcnn_stage1/head/fc7/output:0')
                        fc_2_in = sess.graph.get_tensor_by_name('cascade_rcnn_stage2/multilevel_roi_align/output:0')
                        fc_2_out = sess.graph.get_tensor_by_name('cascade_rcnn_stage2/head/fc7/output:0')
                        fc_3_in = sess.graph.get_tensor_by_name('cascade_rcnn_stage3/multilevel_roi_align/output:0')
                        fc_3_out = sess.graph.get_tensor_by_name('cascade_rcnn_stage3/head/fc7/output:0')
                        #fc_maskrcnn
                        fc_maskrcnn_in = sess.graph.get_tensor_by_name('multilevel_roi_align/output:0')
                        fc_maskrcnn_out = sess.graph.get_tensor_by_name('maskrcnn/fcn3/output:0')
                        time_end = time.time()
                        duration_tf_find_tensor = time_end - time_begin
                        print("[TIME][CPU] find tensor time-consuming: {:.3f}ms"
                            .format(duration_tf_find_tensor * 1000))
                        # DTU RUN backbone
                        time_begin = time.time()
                        dtu_outputs_backbone = []
                        engine_backbone.run(
                            [resized_img], dtu_outputs_backbone,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                        time_end = time.time()
                        duration_dtu_backbone = time_end - time_begin
                        print("[TIME][DTU] backbone time-consuming: {:.3f}ms"
                            .format(duration_dtu_backbone * 1000))
                        backbone_time.append(duration_dtu_backbone * 1000)
                        
                        # CPU RUN fc_1_in
                        time_begin = time.time()
                        foutputs_fc_1 = sess.run(
                                            [
                                                fc_1_in, fpn_box
                                            ],
                                            feed_dict={
                                                # input: resized_img
                                                input_fpn_out_2:dtu_outputs_backbone[0],
                                                input_fpn_out_3:dtu_outputs_backbone[1],
                                                input_fpn_out_4:dtu_outputs_backbone[2],
                                                input_fpn_out_5:dtu_outputs_backbone[3],
                                                Lvl2_Reshape_1:dtu_outputs_backbone[4],
                                                Lvl3_Reshape_1:dtu_outputs_backbone[5],
                                                Lvl4_Reshape_1:dtu_outputs_backbone[6],
                                                Lvl5_Reshape_1:dtu_outputs_backbone[7],
                                                Lvl6_Reshape_1:dtu_outputs_backbone[8],
                                                rpn_trans:dtu_outputs_backbone[9],
                                                rpn1_trans:dtu_outputs_backbone[10],
                                                rpn2_trans:dtu_outputs_backbone[11],
                                                rpn3_trans:dtu_outputs_backbone[12],
                                                rpn4_trans:dtu_outputs_backbone[13],
                                            })
                        time_end = time.time()
                        duration_cpu_fc_1_in = time_end - time_begin
                        print("[TIME][CPU] calc fc_1_in time-consuming: {:.3f}ms"
                            .format(duration_cpu_fc_1_in * 1000))
                        
                        # DTU RUN fc_1_out
                        time_begin = time.time()
                        dtu_outputs_fc_1 = []
                        engine_fc_1.run(
                            [foutputs_fc_1[0]], dtu_outputs_fc_1,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                        time_end = time.time()
                        duration_dtu_fc_1 = time_end - time_begin
                        print("[TIME][DTU] fc_1 time-consuming: {:.3f}ms"
                            .format(duration_dtu_fc_1 * 1000))

                        # CPU RUN fc_2_in
                        time_begin = time.time()
                        foutputs_fc_2 = sess.run(
                                            [
                                                fc_2_in, stage_1_box
                                            ],
                                            feed_dict={
                                                # input: resized_img
                                                input_fpn_out_2:dtu_outputs_backbone[0],
                                                input_fpn_out_3:dtu_outputs_backbone[1],
                                                input_fpn_out_4:dtu_outputs_backbone[2],
                                                input_fpn_out_5:dtu_outputs_backbone[3],
                                                fc_1_out:dtu_outputs_fc_1[0],
                                                fpn_box:foutputs_fc_1[1]
                                            })
                        time_end = time.time()
                        duration_cpu_fc_2_in = time_end - time_begin
                        print("[TIME][CPU] calc fc_2_in time-consuming: {:.3f}ms"
                            .format(duration_cpu_fc_2_in * 1000))
                        
                        # DTU RUN fc_2_out
                        time_begin = time.time()
                        dtu_outputs_fc_2 = []
                        engine_fc_2.run(
                            [foutputs_fc_2[0]], dtu_outputs_fc_2,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                        time_end = time.time()
                        duration_dtu_fc_2 = time_end - time_begin
                        print("[TIME][DTU] fc_2 time-consuming: {:.3f}ms"
                            .format(duration_dtu_fc_2 * 1000))

                        # CPU RUN fc_3_in
                        time_begin = time.time()
                        foutputs_fc_3 = sess.run(
                                            [
                                                fc_3_in, stage_2_box
                                            ],
                                            feed_dict={
                                                # input: resized_img
                                                input_fpn_out_2:dtu_outputs_backbone[0],
                                                input_fpn_out_3:dtu_outputs_backbone[1],
                                                input_fpn_out_4:dtu_outputs_backbone[2],
                                                input_fpn_out_5:dtu_outputs_backbone[3],
                                                fc_2_out:dtu_outputs_fc_2[0],
                                                stage_1_box:foutputs_fc_2[1]
                                            })
                        time_end = time.time()
                        duration_cpu_fc_3_in = time_end - time_begin
                        print("[TIME][CPU] calc fc_3_in time-consuming: {:.3f}ms"
                            .format(duration_cpu_fc_3_in * 1000))

                        # DTU RUN fc_3_out
                        time_begin = time.time()
                        dtu_outputs_fc_3 = []
                        engine_fc_3.run(
                            [foutputs_fc_3[0]], dtu_outputs_fc_3,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                        time_end = time.time()
                        duration_dtu_fc_3 = time_end - time_begin
                        print("[TIME][DTU] fc_3 time-consuming: {:.3f}ms"
                            .format(duration_dtu_fc_3 * 1000))

                        # CPU RUN fc_maskrcnn_in
                        time_begin = time.time()
                        foutputs_fc_maskrcnn = sess.run(
                                            [
                                                fc_maskrcnn_in, boxes, scores, labels, 
                                            ],
                                            feed_dict={
                                                # input: resized_img
                                                input_fpn_out_2:dtu_outputs_backbone[0],
                                                input_fpn_out_3:dtu_outputs_backbone[1],
                                                input_fpn_out_4:dtu_outputs_backbone[2],
                                                input_fpn_out_5:dtu_outputs_backbone[3],
                                                fc_1_out:dtu_outputs_fc_1[0], 
                                                fc_2_out:dtu_outputs_fc_2[0], 
                                                fc_3_out:dtu_outputs_fc_3[0], 
                                                stage_2_box:foutputs_fc_3[1]
                                            })
                        time_end = time.time()
                        duration_cpu_fc_maskrcnn_in = time_end - time_begin
                        print("[TIME][CPU] calc fc_maskrcnn_in time-consuming: {:.3f}ms"
                            .format(duration_cpu_fc_maskrcnn_in * 1000))

                        # DTU RUN fc_maskrcnn_out
                        time_begin = time.time()
                        dtu_outputs_fc_maskrcnn = []
                        engine_fc_maskrcnn.run(
                            [foutputs_fc_maskrcnn[0]], dtu_outputs_fc_maskrcnn,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                        time_end = time.time()
                        duration_dtu_fc_maskrcnn = time_end - time_begin
                        print("[TIME][DTU] fc_maskrcnn time-consuming: {:.3f}ms"
                            .format(duration_dtu_fc_maskrcnn * 1000))
                        
                        # CPU RUN endding
                        time_begin = time.time()
                        foutputs_end = sess.run(
                                            [
                                                masks,
                                            ],
                                            feed_dict={
                                                fc_maskrcnn_out:dtu_outputs_fc_maskrcnn[0],
                                                labels:foutputs_fc_maskrcnn[3]
                                            })
                        time_end = time.time()
                        duration_cpu_endding = time_end - time_begin
                        print("[TIME][CPU] calc endding time-consuming: {:.3f}ms"
                            .format(duration_cpu_endding * 1000))
                        time_total_end = time.time()
                        print("[TIME] TOTAL time-consuming: {:.3f}ms"
                        .format((time_total_end - time_total_begin) * 1000))
                        total_time.append((time_total_end - time_total_begin) * 1000)
                        
                        boxes = foutputs_fc_maskrcnn[1]
                        scores = foutputs_fc_maskrcnn[2]
                        labels = foutputs_fc_maskrcnn[3]
                        masks = foutputs_end[0]
                    print(np.mean(backbone_time[1:]))
                    print(np.mean(total_time[1:]))
                    # for i in range(4):
                    #     print("$$$$"*30)
                    #     print(foutputs[4 + i].shape)

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
