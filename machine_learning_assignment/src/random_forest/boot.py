import pandas as pd 
import numpy as np
import random


###The purpose of this python file is to generate a bootstrap dataset



# INPUT: 1: [[[20, 0, 0, 1, 'T']],[[21, 1, 0, 0, 'F']],[[21, 1, 0, 0, 'F']],[[21, 1, 0, 0, 'F']]]  2:Bootstrapped Dataset 
# A list of the individual rows 
# OUTPUT: 
"""
BOOTSTRAPPED DATASET
  ID  P  R  Q Class
0  22  2  0  0     F
1  22  2  0  0     F
2  20  0  0  1     T
3  21  1  0  0     F

"""


#DataFrame that has randomised selections from previous dataset

def boot(rows,bdf):
	for x in range(len(rows)):
		bdf.loc[x] = random.choice(rows)
	return bdf

def verify_boot(rows,bdf,new_bdf):
		# We are going to verify if the bootstrap is different from the original
	##  If the num(trues) == num(rows) then original dataset = bootstrap
	# df['bag'] = np.where(df['ID'] == bdf['ID'],'True','False')
	check_list = np.unique(bdf['ID'])


	# Check if the bootstrap is the same as the original dataset
	while(len(check_list) >= nrows):
		new_bdf = boot(rows,bdf)

	return new_bdf

#Original dataset
df = pd.read_csv("rajin.csv",header=None,names=['ID','P','R','Q','Class'])

# Bootstrap dataset
# Python pointers are going to change df every time we change df
#bdf = df
bdf = pd.read_csv("rajin.csv",header=None,names=['ID','P','R','Q','Class'])

#number of rows
nrows = df.shape[0]
rows = [df.iloc[x] for x in range(nrows)]


#######   EXAMPLE PRINTS ####################
# print("Number of rows", nrows)
# for row in rows:
# 	print("Row: ", row)
#######   EXAMPLE PRINTS ####################


## Bootstrap the dataset
# We are going to generate random numbers of bootstrap the dataset
new_bdf = boot(rows,bdf)
new_bdf = verify_boot(rows,bdf,new_bdf)
print(new_bdf)


# Example Bootstrap
"""
Bootstrap

0  23  1  1  2     T
1  22  2  0  0     F
2  21  1  0  0     F
3  21  1  0  0     F

"""







