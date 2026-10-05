import numpy as np
import os

os.makedirs('demo_data', exist_ok=True)

# Create fake data
samples = 100
# Data: (samples, bands, channels) = (100, 5, 62)
fake_data = np.random.randn(samples, 5, 62).astype(np.float32)
np.savez_compressed('demo_data/DatasetNoImage.npz', data=fake_data)

# Labels: (samples,)
fake_labels = np.random.randint(0, 3, size=(samples,)).astype(np.int64)
np.savez_compressed('demo_data/LabelsNoImage.npz', label=fake_labels)

# Subjects: (samples,)
fake_subjects = np.ones((samples,), dtype=np.int64)
np.savez_compressed('demo_data/SubjectsNoImage.npz', subject=fake_subjects)

print("Demo data generated.")
