from nas_201_api import NASBench201API as API

api = API('nas_201_api/NAS-Bench-201-v1_0-e61699.pth', verbose=False)
print('Load data - done!')
num = len(api)
for i, arch_str in enumerate(api):
    print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
    break
print(num)
