import os

for i in range(20):
    cmd = 'python mainpro_CK+.py --model VGG19 --bs 32 --lr 0.01 --fold %d' %(i+1)
    os.system(cmd)
print("Train VGG19 ok!")

