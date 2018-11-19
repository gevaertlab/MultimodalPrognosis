
import numpy as np
import pandas as pd
import random
import glob, yaml, json, xmltodict, os

from utils import *

import IPython

data = yaml.load(open(f"{DATA_DIR}/processed/case_files_locs.yaml"))
cases = data.keys()

race_lookup = []
drug_lookup = []
disease_lookup = []

def load_cache(cache_file=FETCH_CACHE):
	global data, cases, race_lookup, drug_lookup, disease_lookup

	data_dict = np.load(cache_file, encoding='latin1')
	data = data_dict["data"].tolist()
	cases = sorted(data.keys())

	load_frac = np.array(["loaded" in data[case] for case in cases]).mean()
	print ("Fraction loaded: ", load_frac)

	race_lookup = data_dict['race_lookup'].tolist()
	drug_lookup = data_dict['drug_lookup'].tolist()
	disease_lookup = data_dict['disease_lookup'].tolist()

load_cache()

def cache():
	data_cached = np.array(data)
	np.savez_compressed(cache_file, data=data_cached, race_lookup=race_lookup, 
			disease_lookup=disease_lookup, drug_lookup=drug_lookup)

def case_list(project="TCGA-SKCM"):
	return [case for case in cases if data[case]["project"] == project]

def load_data_from_path(xml_dict, *args):
	for spec in args:
		try:
			key_matching = [key for key in xml_dict.keys() if spec in key][0]
		except: return False

		xml_dict = xml_dict[key_matching]
	return xml_dict

def clinical_data(case):
	patient_data = load(case)
	keys = ["gender", "race", "age", "disease"]
	if not all(key in patient_data for key in keys): return None
	return np.array([patient_data[key] for key in keys])

def clinical_data_expanded(case):
	patient_data = load(case)
	keys = ["gender"]
	if not all(key in patient_data for key in keys): return None
	keys += ["disease", 'age', 'race', 'histologic_grade', 'pathologic_grade', 'drug']
	return np.array([patient_data.get(key, 0) or 0 for key in keys])

def gender(case):
	patient_data = load(case)
	if "gender" not in patient_data: return None
	return patient_data['gender']

def race(case):
	patient_data = load(case)
	if "race" not in patient_data: return None
	return patient_data['race']

def mirna_data(case):
	patient_data = load(case)
	if "mirna_data" not in patient_data: return None
	return patient_data['mirna_data']

def gene_data(case):
	patient_data = load(case)
	if "gene_data" not in patient_data: return None
	return patient_data['gene_data']

def histology_data(case):
	patient_data = load(case)
	keys = ["histologic_grade", "pathologic_grade"]
	if not all(key in patient_data for key in keys): return None
	return np.array([patient_data[key] for key in keys])

def histologic_grade(case):
	patient_data = load(case)
	if "histologic_grade" not in patient_data: return None
	return patient_data['histologic_grade']

def vital_status(case):
	patient_data = load(case)
	if "vital_status" not in patient_data: return None
	return patient_data['vital_status']

def recurrence(case):
	patient_data = load(case)
	if "recurrence" not in patient_data: return None
	return patient_data['recurrence']

def survival(case):
	raise NotImplementedError()

def cancer_type(case):
	patient_data = load(case)
	return patient_data["disease"]

def days_to_death(case):
	patient_data = load(case)
	if "days_to_death" not in patient_data: return None
	return patient_data['days_to_death']

def slide_regions(case, window_size=500, view_size=1000, num=10, max_num=None, tiling=2):
	slide_files = data[case]['slides']
	slides = [open_slide(slide_file, allow_copy=False) for slide_file in slide_files]

	if None in slides: return None
	if len(slides) == 0: return None
	
	try:
		slide_data = sample_from_slides(slides, window_size=window_size, view_size=view_size, 
			tiling=tiling, num=num, max_num=max_num)
	except:
		return None

	return slide_data

def load(case):
	global num

	if "loaded" in data[case]: return data[case]
	dd = xmltodict.parse(open("data/" + data[case]['clinical_data_file'], 'rb'))

	clin = load_data_from_path(dd, "tcga", "patient", "new_tumor_events", "new_tumor_event_after_initial_treatment", "text")
	if clin: data[case]['recurrence'] = 1 if clin == "YES" else 0

	clin = load_data_from_path(dd, "tcga", "patient", 'vital_status', 'text')
	if clin: data[case]['vital_status'] = 1 if clin.upper() == "ALIVE" else 0

	clin = load_data_from_path(dd, "tcga", "patient", 'days_to_death', 'text')
	if "vital_status" in data[case] and data[case]['vital_status'] == 1: data[case]['days_to_death'] = False
	elif clin: data[case]['days_to_death'] = int(clin)

	clin = load_data_from_path(dd, "tcga", "patient", 'gender', 'text')
	if clin: data[case]['gender'] = 1 if clin == "MALE" else 0

	clin = load_data_from_path(dd, "tcga", "patient", 'race_list', 'race', 'text')
	clin2 = load_data_from_path(dd, "tcga", "patient", 'ethnicity', 'text')
	if clin and clin2:
		if clin2 == "HISPANIC OR LATINO":
			clin = "LATINO"
		if "NATIVE" in clin:
			clin = "ASIAN"

		if clin not in race_lookup: race_lookup.append(clin)
		data[case]['race'] = race_lookup.index(clin)

	clin = load_data_from_path(dd, "tcga", "patient", 'age_at_initial', 'text')
	if clin: data[case]['age'] = int(clin)

	clin = load_data_from_path(dd, "tcga", "patient", 'drugs', 'drug', "drug_name", 'text')
	if clin:
		if clin not in drug_lookup: drug_lookup.append(clin)
		data[case]['drug'] = drug_lookup.index(clin)

	clin = load_data_from_path(dd, "tcga", "patient", 'histologic_grade', 'text')
	
	if clin and clin != "GX": 
		if "High" in clin: clin=3
		elif "1" in clin: clin=0
		elif "2" in clin: clin=1
		elif "3" in clin: clin=2
		elif "4" in clin: clin=3
		else: clin = False

		if clin:
			data[case]['histologic_grade'] = clin

	clin = load_data_from_path(dd, "tcga", "patient", 'stage_event', 'pathologic', 'text')
	
	if clin: 
		clin = clin.count('I')
		data[case]['pathologic_grade'] = clin
	
	clin = load_data_from_path(dd, "tcga", "admin:admin", 'disease_code', 'text')
	if clin:
		if clin not in disease_lookup: disease_lookup.append(clin)
		data[case]['disease'] = disease_lookup.index(clin)

	if "mirna_expression_file" in data[case]:
		gene_data_file = "data/" + data[case]['mirna_expression_file']
		df = pd.read_csv(gene_data_file, sep='\t')
		data[case]['mirna_data'] = np.array(df["reads_per_million_miRNA_mapped"])/10000.0

	if "gene_expression_file" in data[case]:
		gene_data_file = "data/" + data[case]['gene_expression_file']
		df = pd.read_csv(gene_data_file, sep='\t', header=None)
		df = df[~df[0].str.contains("__")]
		data[case]['gene_data'] = np.array(df[1])

	data[case]['loaded'] = True
	return data[case]



if __name__ == "__main__":
	
	for case in random.sample(list(cases), len(cases)):
		cancer = disease_lookup[cancer_type(case)]
		if cancer == 'GBM':
			print (case, cancer)
			print (gender(case))
			print (clinical_data(case))

	# cache()