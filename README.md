# CSCI 5527 Project

<!-- BEGIN portfolio-project-animations -->
## Animated project guide

These self-contained SVG animations explain the project workflow. They are illustrative guides, not recorded experiments or new performance results. Static content remains available when reduced motion is requested.

### Representation and attention

![Representation and attention — illustrative animation](docs/portfolio-animations/facial-expression-recognition.svg)

Conceptual CNN and attention regions, not a real Grad-CAM map or a demonstrated accuracy improvement.

<!-- END portfolio-project-animations -->

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
