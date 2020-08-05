import pandas as pd
import csv
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Downsampling, p1

b = pd.read_csv("1_behavioral.csv")
b_0 = b[b['acc_rec']==0]
b_1 = b[b['acc_rec']==1].head(88)
b_1 = pd.concat([b_0, b_1])

p = pd.read_csv("1_pupil.csv")

#Pupil

p_list = []
b_list = []
participants = [1,2,3,4,5,6,8,9,11,13,14,16,17,18,19,20,22,23,24,25,26,27,28,29,32,34,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68]
for i in participants:
	dfb = pd.read_csv(str(i) + '_'+'behavioral.csv')
	dfp = pd.read_csv(str(i) + '_'+'pupil.csv')
	p_list.append(dfp)
	b_list.append(dfb)
print("success")


"""
p_1 = pd.read_csv("1_pupil.csv")
b_2 = pd.read_csv("2_behavioral.csv")
p_2 = pd.read_csv("2_pupil.csv")

b_3 = pd.read_csv("3_behavioral.csv")
p_3 = pd.read_csv("3_pupil.csv")

b_4 = pd.read_csv("4_behavioral.csv")
p_4 = pd.read_csv("4_pupil.csv")

b_5 = pd.read_csv("5_behavioral.csv")
p_5 = pd.read_csv("5_pupil.csv")

b_6 = pd.read_csv("6_behavioral.csv")
p_6 = pd.read_csv("6_pupil.csv")

b_8 = pd.read_csv("8_behavioral.csv")
p_8 = pd.read_csv("8_pupil.csv")

b_9 = pd.read_csv("9_behavioral.csv")
p_9 = pd.read_csv("9_pupil.csv")

b_11 = pd.read_csv("11_behavioral.csv")
p_11 = pd.read_csv("11_pupil.csv")

b_13 = pd.read_csv("13_behavioral.csv")
p_13 = pd.read_csv("13_pupil.csv")

b_14 = pd.read_csv("14_behavioral.csv")
p_14 = pd.read_csv("14_pupil.csv")

b_16 = pd.read_csv("16_behavioral.csv")
p_16 = pd.read_csv("16_pupil.csv")

b_17 = pd.read_csv("17_behavioral.csv")
p_17 = pd.read_csv("17_pupil.csv")

b_18 = pd.read_csv("18_behavioral.csv")
p_18 = pd.read_csv("18_pupil.csv")

b_19 = pd.read_csv("19_behavioral.csv")
p_19 = pd.read_csv("19_pupil.csv")

b_20 = pd.read_csv("20_behavioral.csv")
p_20 = pd.read_csv("20_pupil.csv")
b_list = [b_1, b_2, b_3, b_4, b_5, b_6, b_8, b_9, b_11, b_13, b_14, b_16, b_17, b_18, b_19, b_20]
p_list = [p_1, p_2, p_3, p_4, p_5, p_6, p_8, p_9, p_11, p_13, p_14, p_16, p_17, p_18, p_19, p_20]
"""



"""
b = pd.read_csv("3_behavioral.csv")
b_0 = b[b['acc_rec']==0]
b_1 = b[b['acc_rec']==1]
print(b_0)
print(b_1)
"""
def get_feature_1_2(trial, start, end, participant):
	# Mean pupil size from s to l seconds
	p = p_list[participant]
	p_trial = p[p['Trial']==trial]
	p_trial_bounded = p_trial[p_trial['Time'] >= start]
	p_trial_bounded = p_trial_bounded[p_trial_bounded['Time'] <= end]
	p_sizes = list(p_trial_bounded['Pupil'])
	for i in range(len(p_sizes)):
		if math.isnan(p_sizes[i]):
			p_sizes[i] = 0
	return sum(p_sizes) / len(p_sizes)
	
def get_feature_3(trial, base, participant):
	# Baseline corrected pupil size before trial onset
	p = p_list[participant]
	p_t = p[p['Trial']==trial]
	p_0 = p_t[p_t['Time'] > -0.009]
	p_0 = p_0[p_0['Time'] < 0.009]
	diff = (list(p_0['Pupil'])[0] - get_feature_1_2(trial, base/10-0.3, base/10+0.3, participant)) / max(get_feature_1_2(trial, base/10-0.3, base/10+0.3, participant), 800)
	
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


def populate_csv_by_base(base):
	with open("percentage_test.csv", "w", newline="") as file:
		fieldnames = ['FEATURE_1', "ACC_REC"]
		writer = csv.writer(file)
		writer.writerow(fieldnames)
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		for participant in range(len(b_list)):
			for trial in list(b_list[participant]['trial_index']):
				feature_1 = get_feature_3(trial,base, participant)
				if math.isnan(feature_1):
					continue
				writer.writerow({
					"FEATURE_1": feature_1,
					"ACC_REC": list(b_list[participant][b_list[participant]['trial_index']==trial]['acc_rec'])[0]
				})
				
offsets = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2, -1.3, -1.4, -1.5]



for offset in offsets:
	D = pd.DataFrame(columns=["X", "Y"])
	for p in range(len(p_list)):
		acc_rec = b_list[p]["acc_rec"]
		X = []
		Y = []
		for i in range(1,228):
			x = get_feature_3(i, offset, p)
			if not math.isnan(x):
				X.append(x)
				Y.append(acc_rec[i-1])
		df = pd.DataFrame({"X": X, "Y": Y})
		D = pd.concat([D, df])
		print(p)
		
	logreg = LogisticRegression(solver="liblinear", random_state=1)
	D0 = D[D["Y"]==0]
	D1 = D[D["Y"]==1]
	D = pd.concat([D0, D1.head(3801)])
	features = list(D["X"])
	label = list(D["Y"])
	D = pd.DataFrame({"X": features, "Y": label})
	X = D[["X"]]
	Y = D["Y"]
	acc = cross_val_score(logreg, X, Y, cv=5, scoring="accuracy").mean()
	print(acc)
	with open("results.csv", "a", newline="") as file:
		fieldnames = ['offset', "accuracy"]
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writerow({
			"offset": offset,
			"accuracy": acc
		})









"""
populate_csv_by_base(-2)

df = pd.read_csv("percentage_test.csv")

df_0 = df[df['ACC_REC'] == 0]
df_1 = df[df['ACC_REC'] == 1]
df = pd.concat([df_0, df_1.head(1010)])
logreg = LogisticRegression(solver="liblinear", random_state=1)

X = df[["FEATURE_1"]]
y = df["ACC_REC"]
print(cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean())
"""
"""
df = pd.read_csv("percentage_test.csv")

df_0 = df[df['ACC_REC'] == 0]
df_1 = df[df['ACC_REC'] == 1]
print(df_0)
print(df_1)
df = pd.concat([df_0, df_1.head(515)])
logreg = LogisticRegression(solver="liblinear", random_state=1)

X = df[["FEATURE_1"]]
y = df["ACC_REC"]
print(cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean())	
"""

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


df = pd.read_csv("features.csv")
logreg = LogisticRegression(solver="liblinear", random_state=1)

X = df[["FEATURE_1", "FEATURE_2", "FEATURE_3"]]
y = df["ACC_REC"]
print(cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean())

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
"""
