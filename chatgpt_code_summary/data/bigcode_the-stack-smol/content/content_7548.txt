'''
Created on Feb 4, 2021

@author: paepcke
'''
import io
import os
import pickle
import tempfile
import unittest

from experiment_manager.neural_net_config import NeuralNetConfig


#from experiment_manager.dottable_config import DottableConfigParser
TEST_ALL = True
#TEST_ALL = False

class NeuralNetConfigTest(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        

    def setUp(self):
        cfg_file = os.path.join(os.path.dirname(__file__), 
                                'dottable_config_tst.cfg')
        self.cfg_arr_file = os.path.join(os.path.dirname(__file__), 
                                         'neural_net_config_arrays_tst.cfg')
        self.config = NeuralNetConfig(cfg_file)
        
        complete_cfg_file = os.path.join(os.path.dirname(__file__), 
                                         '../../tests',
                                         'bird_trainer_tst.cfg')
        self.complete_config = NeuralNetConfig(complete_cfg_file)

    def tearDown(self):
        pass

    # ------------ Tests -----------

    #------------------------------------
    # test_add_section 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_section(self):
        self.config.add_section('FoodleDoodle')
        secs = self.config.sections()
        self.assertIn('FoodleDoodle', secs)
        self.assertEqual(len(self.config.FoodleDoodle), 0)
        self.assertEqual(len(self.config['FoodleDoodle']), 0)
        
        self.config.FoodleDoodle = 10
        self.assertEqual(self.config.FoodleDoodle, 10)
        self.assertEqual(self.config['FoodleDoodle'], 10)
        
    #------------------------------------
    # test_setter_evals 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_setter_evals(self):
        
        # A non-neural-net name:
        self.config.foo = 10
        self.assertEqual(self.config.foo, 10)
        self.assertEqual(self.config['foo'], 10)
        
        # A nn-special parameter:
        self.config.batch_size = 128
        self.assertEqual(self.config.Training.batch_size, 128)
        self.assertEqual(self.config.Training['batch_size'], 128)
        
        self.config.Training.optimizer = 'foo_opt'
        self.assertEqual(self.config.Training.optimizer, 'foo_opt')

    #------------------------------------
    # test_setter_methods
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_setter_methods(self):
        #****self.config.net_name = 'Foobar'
        self.config.set_net_name('Foobar')
        self.assertEqual(self.config.Training.net_name, 'Foobar')
        
        # Wrong type for epoch:
        with self.assertRaises(AssertionError):
            self.config.set_min_epochs('foo')
            
        # min_epoch > max_epoch:
        self.config.set_max_epochs(10)
        with self.assertRaises(AssertionError):
            self.config.set_min_epochs('20')
            
        self.config.set_batch_size(32)
        self.assertEqual(self.config.Training.batch_size, 32)
        with self.assertRaises(AssertionError):
            self.config.set_batch_size(-20)
        self.assertEqual(self.config.Training.batch_size, 32)
        
        with self.assertRaises(AssertionError):
            self.config.set_num_folds(-20)
        
        with self.assertRaises(AssertionError):
            self.config.set_all_procs_log(-20)
        self.config.set_all_procs_log(True)
        self.assertTrue(self.config.Parallelism.all_procs_log)
        
    #------------------------------------
    # test_eq 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_eq(self):
        
        self.assertTrue(self.config == self.config)
        # Copies of a NeuralNetConfig instance
        # shouldn't be (content-wise) equal to
        # the original:
        
        conf_copy = self.config.copy()
        self.assertTrue(conf_copy == self.config)
        
        # But if we add a section to the copy
        # (and not to the original)...:
        conf_copy.add_section('TestSection')
        # ... copy and original should no longer
        # be equal:
        self.assertTrue(conf_copy != self.config)
        
        # Check that TestSection was indeed added
        # to the copy, but not simultaneously to the
        # original (via residually shared data structs):
        
        self.assertEqual(sorted(conf_copy.sections()), 
                         sorted(['Paths', 'Training', 'Parallelism', 'TestSection'])
                                )
        self.assertEqual(sorted(self.config.sections()), 
                         sorted(['Paths', 'Training', 'Parallelism'])
                                )

    #------------------------------------
    # test_to_json_xc_recording 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_to_json_xc_recording(self):

        json_str = self.config.to_json()
        expected = '{"Paths": {"my_path": "/foo/bar.txt", "roots": "[/foo/bar.txt, blue.jpg, 10, 3.14]", "toots": "/foo/bar.txt, blue.jpg, 10, 3.14"}, "Training": {"train_str": "resnet18", "train_int": "5", "train_float": "3.14159", "train_yes": "yes", "train_yescap": "Yes", "train_1": "1", "train_on": "On", "train_no": "no", "train_0": "0", "train_off": "off"}, "Parallelism": {}}'
        self.assertEqual(json_str, expected)
        
        str_stream  = io.StringIO()
        full_stream = self.config.to_json(str_stream)
        expected = full_stream.getvalue()
        self.assertEqual(json_str, expected)
        
        # For writing to file, use a temp file
        # that is destroyed when closed:
        try:
            tmp_file = tempfile.NamedTemporaryFile(dir=self.curr_dir,
                                                   suffix='.json',
                                                   delete=False
                                                   )
            tmp_file.close()
            with open(tmp_file.name, 'w') as fd:
                fd.write('foobar')
            
            with self.assertRaises(FileExistsError):
                self.config.to_json(tmp_file.name)
                
        finally:
            tmp_file.close()
            os.remove(tmp_file.name)
    
        try:
            tmp_file = tempfile.NamedTemporaryFile(dir=self.curr_dir,
                                                   suffix='.json',
                                                   delete=False
                                                   )
            tmp_file.close()
            self.config.to_json(tmp_file.name,
                                check_file_exists=False
                                )

        finally:
            tmp_file.close()
            os.remove(tmp_file.name)

    #------------------------------------
    # test_from_json 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_from_json(self):
        
        json_str = self.config.to_json()
        
        new_inst = NeuralNetConfig.json_loads(json_str)
        self.assertTrue(new_inst == self.config)

    #------------------------------------
    # test_json_human_readable 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_json_human_readable(self):
        
        json_str = self.complete_config.to_json()
        human_str = NeuralNetConfig.json_human_readable(json_str)
        print(human_str)


    #------------------------------------
    # test_arrays
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_arrays(self):
        config = NeuralNetConfig(self.cfg_arr_file)
        arr1 = config.getarray('Training', 'arr1', 'foo')
        expected = ['1', "'foo'", '[10', 'bluebell]']
        self.assertListEqual(arr1, expected)
        
        arr2 = config.getarray('Training', 'arr2', 'foo')
        self.assertEqual(len(arr2),10)
        expected = ['BANAG', 'BBFLG', 'BCMMG', 'BHPAG', 'BHTAG', 
                    'BTSAC', 'BTSAS', 'CCROC', 'CCROS', 'CFPAG'
                    ]
        self.assertListEqual(arr2, expected)

    #------------------------------------
    # test_pickling
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_pickling(self):
        
        config = NeuralNetConfig(self.cfg_arr_file)
        new_config = pickle.loads(pickle.dumps(config))
        self.assertEqual(config, new_config)

# -------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
