import os
import random
from glob import glob
import numpy as np
import configs.config_loader as cfg_loader
from collections import defaultdict

cfg = cfg_loader.get_config()

train_all = []
test_all = []
val_all = []

# parse all input files
print('Finding raw files for preprocessing.')
paths = glob( cfg.data_dir + cfg.input_data_glob)
paths = [os.path.dirname(p) for p in paths]

# sort according to class folders (optional)
if cfg.class_folders is None:
    res = {'single_class': paths}
else:
    class_paths = glob( cfg.data_dir + cfg.class_folders)
    res = defaultdict(list)
    for path in paths:
        for class_path in class_paths:
            if path.startswith(class_path):
                res[class_path].append(path)


for class_path in res.keys():

    all_samples = res[class_path]

    random.shuffle(all_samples)

    # Number of examples
    n_total = len(all_samples)

    if cfg.n_val is not None:
        n_val = cfg.n_val
    else:
        n_val = int(cfg.r_val * n_total)

    if cfg.n_test is not None:
        n_test = cfg.n_test
    else:
        n_test = int(cfg.r_test * n_total)

    if n_total < n_val + n_test:
        print('Error: too few training samples.')
        exit()

    n_train = n_total - n_val - n_test

    assert(n_train >= 0)

    # Select elements
    train_all.extend( all_samples[:n_train])
    val_all.extend( all_samples[n_train:n_train+n_val])
    test_all.extend( all_samples[n_train+n_val:])


np.savez(cfg.data_dir + f'/../split_{cfg.exp_name}.npz', train = train_all, test = test_all, val = val_all)