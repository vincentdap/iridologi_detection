from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf
from distutils.version import StrictVersion
import os
import sys
import numpy as np
from PyQt5 import QtCore


class Detection_Iridologi():
    def __init__(self):
        super(Detection_Iridologi, self).__init__()


    def detection_iridologi(self,imgread, imgreal):
        sys.path.append("..")
        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
        MODEL_NAME = 'Detection\iridologi_model'
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join('Detection\data', 'object-detection_iridologi.pbtxt')
        NUM_CLASSES = 2
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        def load_image_into_numpy_array(image):
            last_axis = -1
            dim_to_repeat = 2
            repeats = 3
            grscale_img_3dims = np.expand_dims(image, last_axis)
            training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
            assert len(training_image.shape) == 3
            assert training_image.shape[-1] == 3
            return training_image

        # PATH_TO_TEST_IMAGES_DIR = 'image_uji'
        # TEST_IMAGE_PATHS = [ os.path.join('image_uji', 'image{}.jpg'.format(i)) for i in range(1, 2) ]

        # Size, in inches, of the output images.

        IMAGE_SIZE = (12, 8)

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # image = Image.open('image_uji/image2.jpg')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(imgread)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                image_target = load_image_into_numpy_array(imgreal)
                try:
                    read = ([category_index.get(value).get('id') for index, value in enumerate(classes[0]) if
                             scores[0, index] > .4])
                    self.dc = read[0]
                except IndexError:
                    self.dc = 0
                print(self.dc)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    image_target=image_target,
                    max_boxes_to_draw=2,
                    use_normalized_coordinates=True,
                    line_thickness=4)
                return image_target