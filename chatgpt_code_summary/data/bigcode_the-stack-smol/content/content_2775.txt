import unittest

from facial_recog.app import *
from .test_config import test_run_count, seed, success_perc
from .test_util import *


class TestFR(unittest.TestCase):
    subject_names = dict()
    subject_classes = dict()

    def setUp(self):
        random.seed(seed)

        create_app_dirs()
        setup_logger()

        logging.debug('Seed is %s', seed)

        # only for super strict testing
        # clear_fdb()
        prepare_fdb()

        self.subject_names, self.subject_classes = create_sample()
        logging.info('Subject names: %s', self.subject_names)
        logging.info('Subject classes are: %s', self.subject_classes)

        recreate_db()
        populate_db(self.subject_classes)
        logging.info('New db created')

        clear_dataset()
        copy_dataset(subject_names=self.subject_names)
        logging.info('Training Dataset created')

        clear_recognizers()

        for class_id in get_all_classes():
            train(class_id=class_id)
        logging.info('Classifiers trained')

    def test_fr(self):
        success = 0
        for _ in range(test_run_count):
            random_class = random.choice(get_all_classes())
            random_subject = random.choice(get_class_subjects(random_class))
            random_image = random.choice(
                get_images_for_subject(subject_name=self.subject_names[random_subject]))

            logging.info('Testing subject %s in class %s with image %s', random_subject, random_class, random_image)

            if predict(img=path_to_img(random_image), class_id=random_class) == random_subject:
                success += 1
                logging.info('Test success')
            else:
                logging.warning('Test failed')

        self.assertGreaterEqual(success, int(success_perc * test_run_count))
