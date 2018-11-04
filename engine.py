import subprocess

# subprocess.run(['python', 'visualize.py'])

subprocess.call(['python', 'mainpro_FER.py', '--model VGG19', '--bs', '128', '--lr', '0.01'])

# subprocess.run(['python', 'preprocess_fer2013.py'])