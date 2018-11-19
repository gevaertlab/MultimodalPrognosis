
import numpy as np
import matplotlib.pyplot as plt

import os, sys, random, yaml, shutil, time, glob
import IPython

import openslide
from scipy.misc import imresize


def open_slide(slide_image_path):
	return openslide.open_slide(slide_image_path)


def sample_from_slides(slides, window_size=400, view_size=100, num=10):
	
	samples = []
	while len(samples) < num:
		slide = random.choice(slides)
		XMAX, YMAX = slide.dimensions[0], slide.dimensions[1]

		xv, yv = random.randint(0, XMAX - window_size), random.randint(0, YMAX - window_size)
		window = np.array(slide.read_region((xv, yv), 0, (window_size, window_size)))
		
		if np.array(window).mean() > 200:
			continue

		if np.array(window[:, :, 0]).mean() < 50:
			continue

		if np.array(window[:, :, 2]).mean() > 160:
			continue
		
		window = imresize(window, (view_size, view_size, 4))
		#plt.imshow(window); plt.show()
		samples.append(window)

	return np.array(samples)


def sample_from_patient(case, window_size=400, view_size=100, num=10):
	slide_files = glob.glob(f"/Volumes/Seagate Backup Plus Drive/tissue-slides/{case}*.svs")
	slides = [open_slide(file) for file in slide_files]
	if len(slides) == 0: return None
	return sample_from_slides(slides, window_size=window_size, view_size=view_size, num=num)


if __name__ == "__main__":

	data = yaml.load(open(f"data/processed/case_files_locs.yaml"))
	cases = data.keys()

	rois = [sample_from_patient(case, num=4) for case in random.sample(cases, 10)]
	print (len(rois))
	rois = [roi_set for roi_set in rois if roi_set is not None]
	rois = np.array([roi for roi_set in rois for roi in roi_set])

	import torch
	from torchvision.utils import make_grid
	from torchvision import transforms

	print (rois.shape)
	rois = torch.tensor(rois).permute((0, 3, 1, 2))[:, 0:3]
	print (rois.shape)

	transform = transforms.Compose([
									transforms.ToPILImage(),
									transforms.Resize(64),
									transforms.ToTensor()])
	rois = torch.stack([transform(x) for x in rois.cpu()])
	print (rois.shape)
	
	image = make_grid(rois, nrow=4)
	print (image.shape)

	plt.imsave("results/rois.png", image.permute((1, 2, 0)).data.cpu().numpy())
	