import os
import sys
import tensorflow as tf
import shutil


def export(model, sess, signature_name, export_path, version=1):
    # export path
    export_path = os.path.join(os.path.realpath(export_path), signature_name, str(version))
    # print("export_path:{}".format(export_path))
    # 'export_path:/home/yanghk/ner/cls_ner_align_merg/export_model/ner/1'
    if os.path.isdir(export_path):
        shutil.rmtree(export_path)
    print('Exporting trained model to {} ...'.format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    sequencelabel_w = tf.saved_model.utils.build_tensor_info(model.char_inputs)
    sequencelabel_seg = tf.saved_model.utils.build_tensor_info(model.seg_inputs)
    sequencelabel_dropout = tf.saved_model.utils.build_tensor_info(model.dropout)
    sequencelabel_target = tf.saved_model.utils.build_tensor_info(model.targets)

    sequencelabel_trans = tf.saved_model.utils.build_tensor_info(model.trans)
    sequencelabel_lengths = tf.saved_model.utils.build_tensor_info(model.lengths)
    sequencelabel_scores = tf.saved_model.utils.build_tensor_info(model.logits)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs = {'input_w': sequencelabel_w,
                  'input_seg': sequencelabel_seg,
                   'dropout': sequencelabel_dropout,
                  'target': sequencelabel_target
                   },
        outputs = {'trans': sequencelabel_trans,
                   'lengths': sequencelabel_lengths,
                   'scores': sequencelabel_scores
                   },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_name: prediction_signature,
        })
    builder.save()
