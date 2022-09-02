#!/bin/bash
python3 -u code/train_ae.py
python3 -u code/generate_ld_dataset.py
python3 -u code/train_don.py
