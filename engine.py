import subprocess

# subprocess.run(['python', 'visualize.py'])

subprocess.call(['python', 'plot_fer2013_confusion_matrix_cpu.py', '--model', 'VGG19', '--split', 'PrivateTest'])

# subprocess.run(['python', 'preprocess_fer2013.py'])