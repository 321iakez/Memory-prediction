import pandas as pd
import csv
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Downsampling
b = pd.read_csv("1_behavioral.csv")
b_0 = b[b['acc_rec']==0]
b_1 = b[b['acc_rec']==1].head(88)
b = pd.concat([b_0, b_1])

#Pupil
p = pd.read_csv("1_pupil.csv")

#Eyetracking
e = pd.read_csv("1_eyetracking.csv")



def get_feature_1_2(trial, start, end):
	# Mean pupil size from s to l seconds
	p_trial = p[p['Trial']==trial]
	p_trial_bounded = p_trial[p_trial['Time'] >= start]
	p_trial_bounded = p_trial_bounded[p_trial_bounded['Time'] <= end]
	p_sizes = list(p_trial_bounded['Pupil'])
	for i in range(len(p_sizes)):
		if math.isnan(p_sizes[i]):
			p_sizes[i] = 0
	return sum(p_sizes) / len(p_sizes)
	
def get_feature_3(trial):
	# Baseline corrected pupil size before trial onset
	p_t = p[p['Trial']==trial]
	p_0 = p_t[p_t['Time'] > -0.009]
	p_0 = p_0[p_0['Time'] < 0.009]
	print(list(p_0['Pupil'])[0])
	diff = list(p_0['Pupil'])[0] - get_feature_1_2(trial, -1, 0)
	
	return diff

def get_feature_4(trial):
	# Reaction time
	rt_list = list(b['rt'])
	for i in range(len(rt_list)):
		if math.isnan(rt_list[i]):
			rt_list[i] = 0
	mean_rt = sum(rt_list) / len(rt_list)
	b_trial = b[b['trial_index']==trial]
	rt = list(b_trial['rt'])[0]
	if math.isnan(rt):
		return mean_rt
	else:
		return rt
	

def get_feature_5(trial):
	# Number of fixations on a stimulus
	e_trial = e[e['TRIAL_INDEX']==trial]
	fixation_indices = list(e_trial['RIGHT_FIX_INDEX'])
	return int(max(fixation_indices))

def get_feature_6(trial):
	# Mean length of fixations on a stimulus
	num_fixations = get_feature_5(trial)
	total_fixation = 0
	e_trial = e[e['TRIAL_INDEX']==trial]
	for i in range(1,num_fixations+1):
		e_fixation = e_trial[e_trial['RIGHT_FIX_INDEX']==str(i)]
		total_fixation += e_fixation.shape[0]
	return total_fixation/num_fixations/1000


def get_feature_7(trial, n):
	# Number of blinks n seconds after stimulus presentation
	e_trial = e[e['TRIAL_INDEX']==trial]
	e_display_index = list(e_trial[e_trial['SAMPLE_MESSAGE']=='display']['SAMPLE_INDEX'])[0]
	e_trial_bounded = e_trial[e_trial['SAMPLE_INDEX'] >= e_display_index]
	e_trial_bounded = e_trial_bounded[e_trial_bounded['SAMPLE_INDEX'] <= e_display_index+n*1000]
	e_blinks_list = list(e_trial_bounded['RIGHT_IN_BLINK'])
	
	num_blinks = 0
	for i in range(1, len(e_blinks_list)):
		if e_blinks_list[i] == 1 and e_blinks_list[i-1] == 0:
			num_blinks += 1
	return num_blinks

"""
with open("features.csv", "a", newline="") as file:
	fieldnames = ['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5', 'FEATURE_6', 'FEATURE_7', "ACC_REC"]
	writer = csv.DictWriter(file, fieldnames=fieldnames)
	for i in list(b['trial_index']):
		writer.writerow({
			"FEATURE_1": get_feature_1_2(i, -1, 0),
			"FEATURE_2": get_feature_1_2(i, 0, 1),
			"FEATURE_3": get_feature_3(i),
			"FEATURE_4": get_feature_4(i),
			"FEATURE_5": get_feature_5(i),
			"FEATURE_6": get_feature_6(i),
			"FEATURE_7": get_feature_7(i, 3),
			"ACC_REC": list(b[b['trial_index']==i]['acc_rec'])[0]
		})
"""

df = pd.read_csv("features.csv")
logreg = LogisticRegression(solver="liblinear", random_state=1)
"""
X = df[["FEATURE_1", "FEATURE_2", "FEATURE_3"]]
y = df["ACC_REC"]
print(cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean())
"""
features = ["FEATURE_1", "FEATURE_2", "FEATURE_3", "FEATURE_4", "FEATURE_5", "FEATURE_6", "FEATURE_7"]

for i1 in range(2):
	for i2 in range(2):
		for i3 in range(2):
			for i4 in range(2):
				for i5 in range(2):
					for i6 in range(2):
						for i7 in range(2):
							trial_features = []
							if(i1):
								trial_features.append(features[0])
							if(i2):
								trial_features.append(features[1])
							if(i3):
								trial_features.append(features[2])
							if(i4):
								trial_features.append(features[3])
							if(i5):
								trial_features.append(features[4])
							if(i6):
								trial_features.append(features[5])
							if(i7):
								trial_features.append(features[6])
							X = df[trial_features]
							y = df["ACC_REC"]
							print(cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean())

