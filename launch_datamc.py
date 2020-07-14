for subprocess import Popen

sets = {"on", "sideband",'set1','set2','set3','set4','set5'}

for process in sets:
    command = ['python', 'gpvm_datamc_{}.py'.format(process)]
    log = open('./output/nue/datamc/{}.txt'.format(process), 'w')
    result = Popen(command, stdout=log, stderr=log)