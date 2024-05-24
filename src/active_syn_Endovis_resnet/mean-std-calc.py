import os
import numpy as np
import re


def get_numbers(txt_file):
    
    file = open(txt_file, 'r')
    line=file.readline()
    numbers = re.findall(r"\d+\.?\d*",line)
    file.close()
    return numbers
    
ious=[]
dices=[]
maious=[]
marses=[]

for subfix in ['-c1','-c2','-c3']:
    dice,iou = get_numbers('test_rst{}.txt'.format(subfix))
    maiou,marse=get_numbers('test_multi_angle{}.txt'.format(subfix))
    ious.append(float(iou))
    dices.append(float(dice))
    maious.append(float(maiou))
    marses.append(float(marse))


#ious=[0.7567280488089588,0.8109193417862222,0.831311552939287]
#dices=[0.8245355461760062,0.8616966535647888,0.8834150685635438]
#maious=[0.6907888132848551,0.7125759087032473,0.7132675581650518]
#marses=[0.24911057006769655,0.21808803502095114,0.21925637865709763]


file = open('final-rsts.txt', 'w')
file.write('mDSC='+str(np.mean(dices))+',mIOU='+str(np.mean(ious)))
file.write('\n')
file.write('mMA-IOU='+str(np.mean(maious))+',mMA-RSE='+str(np.mean(marses)))
file.write('\n')
file.write('std-mDSC='+str(np.std(dices))+',std-mIOU='+str(np.std(ious)))
file.write('\n')
file.write('std-mMA-IOU='+str(np.std(maious))+',std-mMA-RSE='+str(np.std(marses)))
file.write('\n')
file.close()



