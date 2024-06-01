import logging as log
import cv2
import sys
import numpy as np


class LandmarksDetectionModel:
    '''
    Class for the Face Landmarks Detection Model.

    Load and configure inference plugins for the specified target devices,
    and performs either synchronous or asynchronous modes for the
    specified infer requests.
    '''

    def __init__(self, model_name, device='CPU', extensions=None, async_infer=True):
        '''
        Set instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.infer_request_handle = None

        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.async_infer = async_infer

    def load_model(self, plugin):
        '''
        This method is for loading the model (in IR format) to the device specified by the user.
        Default device is CPU.
        '''

        # Get model
        model_structure = self.model_name + '.xml'
        model_weights = self.model_name + '.bin'

        # Initialize the plugin - load the inference engine API
        # Plugin is the one already created for the Face Detection model
        self.plugin = plugin

        # Add a CPU extension, if applicable
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        # Read the IR as IENetwork
        try:
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except:
            raise ValueError("Could not initialise the network. Have you entered the correct model path?")

        # Check if model and CPU plugin are supported
        if self.device == 'CPU':
            self.check_model()

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)

        # Get the input and output layers
        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        return

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        if np.all(np.array(image.shape)):
            # Create input image to feed into the network
            net_input = {self.input_blob: self.preprocess_input(image)}

            # Start inference. Infer mode (async/sync) is input by user
            if self.async_infer:
                self.infer_request_handle = self.exec_network.start_async(request_id=0, inputs=net_input)
                # Wait for the result of the inference
                if self.exec_network.requests[0].wait(-1) == 0:
                    # Get result of the inference request
                    outputs = self.infer_request_handle.outputs[self.output_blob]
                    eyes_coords, crop_left, crop_right = self.preprocess_output(outputs, image)

            else:
                self.infer_request_handle = self.exec_network.infer(inputs=net_input)
                # Get result of the inference request
                outputs = self.infer_request_handle[self.output_blob]
                eyes_coords, crop_left, crop_right = self.preprocess_output(outputs, image)

        else:
            eyes_coords = []
            crop_left = []
            crop_right = []

        return eyes_coords, crop_left, crop_right

    def check_model(self):
        '''
        This method check whether the model (along with the plugin) is support on the CPU device.
        If anything is missing (such as a CPU extension), let the user know and exit the programm.
        '''

        supported_layers = self.plugin.query_network(network=self.network, device_name='CPU')
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            if self.extensions:
                log.error("The extensions specified do not support some layers. Please specify a new extension.")
            else:
                log.error(
                    "Please try to specify an extension library path by using the --extensions command line argument.")
            sys.exit(1)
        return

    def preprocess_input(self, image):
        '''
        Method to process inputs before feeding them into the model for inference.
        '''
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs, image):
        '''
        Method to process outputs before feeding them into the next model for
        inference or for the last step of the app.
        '''

        w = image.shape[1]
        h = image.shape[0]
        outputs = outputs[0]

        xl, yl = int(outputs[0][0][0] * w), int(outputs[1][0][0] * h)
        xr, yr = int(outputs[2][0][0] * w), int(outputs[3][0][0] * h)

        eyes_coords = [xl, yl, xr, yr]

        # Using the fact that eyes take 1/5 of your face width
        # define bounding boxes around the eyes according to this
        square_size = int(w / 10)
        left_eye_box = [xl - square_size, yl - square_size, xl + square_size, yl + square_size]
        right_eye_box = [xr - square_size, yr - square_size, xr + square_size, yr + square_size]

        crop_left = image[left_eye_box[1]:left_eye_box[3], left_eye_box[0]:left_eye_box[2]]
        crop_right = image[right_eye_box[1]:right_eye_box[3], right_eye_box[0]:right_eye_box[2]]

        return eyes_coords, crop_left, crop_right
