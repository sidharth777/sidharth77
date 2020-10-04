import tensorflow as tf
import nbformat
import pytest

from ipynb.fs.full.quick_draw import create_model_architecture


class TestModel:
    @pytest.fixture(autouse=True)
    def get_model(self):
        model = create_model_architecture()
        self.model_config = model.get_config()
    def test_nn(self):
        #layer0
        assert self.model_config['layers'][0]['class_name'] == 'Conv2D'
        assert self.model_config['layers'][0]['config']['filters'] == 16
        assert self.model_config['layers'][0]['config']['kernel_size'] == (3,3)
        assert self.model_config['layers'][0]['config']['activation'] == 'relu'
        assert self.model_config['layers'][0]['config']['batch_input_shape'] == (None, 28, 28, 1)

        #layer1
        assert self.model_config['layers'][1]['class_name'] == 'MaxPooling2D'
        assert self.model_config['layers'][1]['config']['pool_size'] == (2,2)

        #layer2
        assert self.model_config['layers'][2]['class_name'] == 'Conv2D'
        assert self.model_config['layers'][2]['config']['filters'] == 32
        assert self.model_config['layers'][2]['config']['kernel_size'] == (3,3)
        assert self.model_config['layers'][2]['config']['activation'] == 'relu'
        #layer3
        assert self.model_config['layers'][3]['class_name'] == 'MaxPooling2D'
        assert self.model_config['layers'][3]['config']['pool_size'] == (2,2)
        #layer4
        assert self.model_config['layers'][4]['class_name'] == 'Conv2D'
        assert self.model_config['layers'][4]['config']['filters'] == 64
        assert self.model_config['layers'][4]['config']['kernel_size'] == (3,3)
        assert self.model_config['layers'][4]['config']['activation'] == 'relu'
        #layer5
        assert self.model_config['layers'][5]['class_name'] == 'MaxPooling2D'
        assert self.model_config['layers'][5]['config']['pool_size'] == (2,2) 

        #layer6
        assert self.model_config['layers'][6]['class_name'] == 'Flatten'
        #layer7
        assert self.model_config['layers'][7]['class_name'] == 'Dense'
        assert self.model_config['layers'][7]['config']['units'] == 128
        assert self.model_config['layers'][7]['config']['activation'] == 'relu'
        #layer8
        assert self.model_config['layers'][8]['class_name'] == 'Dense'
        assert self.model_config['layers'][8]['config']['activation'] == 'softmax'
        assert self.model_config['layers'][8]['config']['units'] == 2


    
                                                                                                            
