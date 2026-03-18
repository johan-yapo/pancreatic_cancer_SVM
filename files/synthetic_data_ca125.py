import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

RNG_SEED = 8
np.random.seed(RNG_SEED)
curr_dir = os.getcwd()
DATA_PATH = os.path.join(os.path.dirname(curr_dir), 'Data')
RESULTS_PATH = os.path.join(os.path.dirname(curr_dir), 'Results')

# Load real CA-125 dataset
real_data = pd.read_csv(os.path.join(DATA_PATH, "machine_learning_data_template_ca125 serum.csv"))

TARGET = "Class_Multimodal"  

# Define metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)