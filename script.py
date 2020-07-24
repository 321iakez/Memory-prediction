import pandas as pd
import csv
import math

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
	print(e_blinks_list)
	return 0

get_feature_7(44,3)
