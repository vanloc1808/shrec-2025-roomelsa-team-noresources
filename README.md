# Team `NoResources` at the ROOMELSA Challenge at SHREC 2025

This is the solution for our team (`NoResources`) at the ROOMELSA Challenge at SHREC 2025. The challenge is for the retrieval of optimal objects for multi-modal enhanced language and spatial assistance. The challenge is organized by the [SHREC](https://www.shrec.net/) team.

## Solution

The technical report will be linked later.

## Running the code

### Environment setup

First, please create a separate virtual environment for the project.
You can use [`conda`](https://www.anaconda.com/), [`venv`](https://docs.python.org/3/library/venv.html), or [`uv`](https://docs.astral.sh/uv/) to create the environment.  
It is recommended to use Python 3.11 or later.

```bash
# Create a new conda environment
conda create -n shrec2025 python=3.11
# Activate the environment
conda activate shrec2025
```

Then, install the required packages.

```bash
# Install the required packages
pip install -r requirements.txt
```

### Running the code

First of all, you need to run the mask extraction process.  
Please make sure to change `SCENE_DIR` constant in the `extract_mask.py` to the directory contain scenes files (in my cases, it is `data/private/scenes`).  
Then, run the following command to extract the masks.

```bash
python extract_mask.py
```

After that, run the main script for retrieval.  
Please make sure to change `SCENE_DIR` constant in the `run.py` to the directory containing scenes files (in my cases, it is `data/private/scenes`), and `OBJECT_DIR` to the directory containing object files (in my cases, it is `data/private/objects`).  
Then, run the following command to run the retrieval.

```bash
python run.py
```

Results will be saved to CSV files.
