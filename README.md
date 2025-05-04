# CPS4801 Final Project: COVID-19 Chest X-ray Classification

**Author**: Sun Xubin  
**Course**: CPS 4801 (Spring 2025)

## Project Overview
We fine-tune a ResNet-50 model to classify chest X-ray images into  
*No Infection Sign* and *Limited Infection Sign*.  
Key techniques: Focal Loss, data augmentation, 5-fold CV, OneCycleLR.

## Folder Structure
| Folder | Contents |
|--------|----------|
| `notebook/` | Google Colab notebook |
| `src/` | Python scripts (`train.py`, `utils.py`) |
| `figures/` | Result charts (PNG) |
| `report/` | IEEE Word template & final PDF |

## Quick Start
```bash
pip install -r requirements.txt
python src/train.py
