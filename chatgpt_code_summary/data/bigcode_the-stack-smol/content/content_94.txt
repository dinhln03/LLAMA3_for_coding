import sys
sys.path.append('../scripts')
from detect_duplicates import df

def test_nan_names():
    assert df.name.isnull().sum() == 0
    
def test_dup_pid():
    assert df.patient_id.duplicated().sum() == 0

def test_phone_dup():
    assert df.phone_number.duplicated().sum() == 0