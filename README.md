# CSCI 5527 Project

## Colab Instructions

### Getting the Data
To use the FER2013 data:
```python
import kagglehub

path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)
```

### Using `dataset.py`

In Colab, run the following:
```bash
!git clone https://github.com/RohitPoduval1/csci5527-project.git
```

```python
import sys

sys.path.append('/content/csci5527-project')

# Import FERDataset and use like a normal Dataset
from fer_dataset import FERDataset
```
