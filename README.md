# Deep Learning with Multimodal Representation for Pancancer Prognosis Prediction
Our model estimates the time-til-death for patients across 20 different cancer types using the vast amount of multimodal data that is available for cancer patients. 
We developed an unsupervised encoder to compress these four data modalities into a patient feature vectors, using deep highway networks to extract features from clinical and genomic data, and dilated convolutional neural networks to extract features from whole slide images. These feature encodings were then used to predict single cancer and pancancer overall survival, achieving a C-index of 0.78 overall.
Our model handles multiple data modalities, efficiently analyzes WSIs, and represents patient multi-modal data flexibly into an unsupervised, informative representation resilient to noise and missing data.

*For more details, see [our paper](https://www.biorxiv.org/content/10.1101/577197v1).* 

## Installation
```
git clone https://github.com/gevaertlab/MultimodalPrognosis
cd MultimodalPrognosis
pip install -r requirements.txt
```

## Running Experiments
All experiments are in the `MultimodalPrognosis/experiments` directory. 

 `plot.py` is a good visualization of the performance of the complete multimodal network (run on all the data, with multimodal dropout). 

If you’d like to run experiments with only subsets of the data (e.g only clinical and gene expression data) use the `chartX.py` files.
* `chart1.py` — miRNA and clinical data
* `chart2.py` — gene expression and clinical data
* `chart3.py` — miRNA, gene expression and clinical data
* `chart4.py` — miRNA, slide and clinical data
* `chart5.py` — miRNA, gene expression, slide and clinical data

To run the experiment of your choice, simply type `python  experiments/chartX.py multi`, with `multi` specifying the use of [multimodal dropout](https://www.biorxiv.org/content/10.1101/577197v1?rss=1).
To run the experiment without multimodal dropout, do not include `multi`.

_Note: This code is built to run on a CPU._

