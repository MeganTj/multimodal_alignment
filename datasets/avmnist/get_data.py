import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import pdb

#data dir is the avmnist folder
def get_dataloader(data_dir,batch_size=40,num_workers=8,train_shuffle=True,flatten_audio=False,flatten_image=False,unsqueeze_channel=True,generate_sample=False,normalize_image=True,normalize_audio=True,no_robust=False,to4by4=False,fracs=1):
    trains=[np.load(data_dir+"/image/train_data.npy"),np.load(data_dir+"/audio/train_data.npy"),np.load(data_dir+"/train_labels.npy")]
    tests=[np.load(data_dir+"/image/test_data.npy"),np.load(data_dir+"/audio/test_data.npy"),np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[1]=trains[1].reshape(60000,112*112)
        tests[1]=tests[1].reshape(10000,112*112)
    if generate_sample:
        saveimg(trains[0][0:100])
        saveaudio(trains[1][0:9].reshape(9,112*112))
    trains[0] = np.float32(trains[0])
    tests[0] = np.float32(tests[0])
    trains[1] = np.float32(trains[1])
    tests[1] = np.float32(tests[1])
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        trains[1] = trains[1]/255.0
        tests[1] = tests[1]/255.0
    if not flatten_image:
        trains[0]=trains[0].reshape(60000,28,28)
        tests[0]=tests[0].reshape(10000,28,28)
    if to4by4:
        trains[0]=to4by4img(trains[0])
        tests[0]=to4by4img(tests[0])
        trains[1]=to4by4aud(trains[1])
        tests[1]=to4by4aud(tests[1])
    if unsqueeze_channel:
        trains[0]=np.expand_dims(trains[0],1)
        tests[0]=np.expand_dims(tests[0],1)
        trains[1]=np.expand_dims(trains[1],1)
        tests[1]=np.expand_dims(tests[1],1)
    trains[2]=trains[2].astype(int)
    tests[2]=tests[2].astype(int)
    trainlist=[[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist=[[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000],shuffle=False,num_workers=num_workers,batch_size=batch_size)
    tests = DataLoader(testlist,shuffle=False,num_workers=num_workers,batch_size=batch_size)
    trains = DataLoader(trainlist[0:int(55000*fracs)],shuffle=train_shuffle,num_workers=num_workers,batch_size=batch_size)
    return trains,valids,tests

def get_dataloader_mfcc(data_dir,batch_size=40,num_workers=8,train_shuffle=True,flatten_audio=False,flatten_image=False,unsqueeze_channel=True,generate_sample=False,normalize_image=True,normalize_audio=True,
                        no_robust=False,to4by4_img=False, to4by4_aud=False, fracs=1):
    trains=[np.load(data_dir+"/image/train_data.npy"),np.load(data_dir+"/audio/train_data.npy"),np.load(data_dir+"/train_labels.npy")]
    tests=[np.load(data_dir+"/image/test_data.npy"),np.load(data_dir+"/audio/test_data.npy"),np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[1]=trains[1].reshape(60000,112*112)
        tests[1]=tests[1].reshape(10000,112*112)
    if generate_sample:
        saveimg(trains[0][0:100])
        saveaudio(trains[1][0:9].reshape(9,112*112))
    trains[0] = np.float32(trains[0])
    tests[0] = np.float32(tests[0])
    trains[1] = np.float32(trains[1])
    tests[1] = np.float32(tests[1])
    if normalize_image:
        trains[0] /= 255.0
        tests[0] /= 255.0
    if normalize_audio:
        # Get train stats for each mfcc 
        # trains[1] is (num_data, num_mfcc, num timeframes)
        max_train_idx = int(55000*fracs)
        train_mean, train_std = trains[1][:max_train_idx].mean(axis=(0, 2)), trains[1][:max_train_idx].std(axis=(0, 2))
        train_mean = train_mean.reshape(1, len(train_mean), 1)
        train_std = train_std.reshape(1, len(train_std), 1)
        trains[1] = (trains[1] - train_mean) / train_std
        tests[1] = (tests[1] - train_mean) /train_std
    if not flatten_image:
        trains[0]=trains[0].reshape(60000,28,28)
        tests[0]=tests[0].reshape(10000,28,28)
    if to4by4_img:
        trains[0]=to4by4img(trains[0])
        tests[0]=to4by4img(tests[0])
    if to4by4_aud:
        trains[1]=to4by4audmfcc(trains[1])
        tests[1]=to4by4audmfcc(tests[1])
    if unsqueeze_channel:
        trains[0]=np.expand_dims(trains[0],1)
        tests[0]=np.expand_dims(tests[0],1)
        if to4by4_aud:
            trains[1]=np.expand_dims(trains[1],1)
            tests[1]=np.expand_dims(tests[1],1)
    trains[2]=trains[2].astype(int)
    tests[2]=tests[2].astype(int)
    trainlist=[[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist=[[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000],shuffle=False,num_workers=num_workers,batch_size=batch_size)
    tests = DataLoader(testlist,shuffle=False,num_workers=num_workers,batch_size=batch_size)
    trains = DataLoader(trainlist[0:max_train_idx],shuffle=train_shuffle,num_workers=num_workers,batch_size=batch_size)
    return trains,valids,tests

def to4by4img(inp):
    batch = inp.shape[0]
    b = []
    for img in inp:
        for i in range(7):
            for j in range(7):
                b.append(img[i*4:i*4+4,j*4:j*4+4].flatten())
    b = np.array(b).reshape(batch,7,7,16)
    return b

    
def to4by4aud(inp):
    batch = inp.shape[0]
    # Input is either 112 or 64
    num_patch_dim = inp.shape[1] // 16
    b=[]
    for img in inp:
        for i in range(num_patch_dim):
            for j in range(num_patch_dim):
                b.append(img[i*16:i*16+16,j*16:j*16+16].flatten())
    b = np.array(b).reshape(batch,num_patch_dim,num_patch_dim,256)
    return b

def to4by4audmfcc(inp):
    batch = inp.shape[0]
    # Input is either 112 or 64
    num_patch_dim = [inp.shape[1] // 4, inp.shape[2] // 4]
    b=[]
    for img in inp:
        for i in range(num_patch_dim[0]):
            for j in range(num_patch_dim[1]):
                b.append(img[i*4:i*4+4,j*4:j*4+4].flatten())
    b = np.array(b).reshape(batch,num_patch_dim[0],num_patch_dim[1],16)
    return b
    
   

# this function creates an image of 100 numbers in avmnist
def saveimg(outa):
    from PIL import Image
    t = np.zeros((300,300))
    for i in range(0,100):
        for j in range (0,784):
            imrow = i // 10
            imcol = i % 10
            pixrow = j // 28
            pixcol = j % 28
            t[imrow*30+pixrow][imcol*30+pixcol]=outa[i][j]
    newimage = Image.new('L', (300, 300))  # type, size
    # print(t)
    newimage.putdata(t.reshape((90000,)))
    newimage.save("samples.png")
def saveaudio(outa):
    #print(outa.shape)
    from PIL import Image
    t = np.zeros((340,340))
    for i in range(0,9):
        for j in range (0,112*112):
            imrow = i // 3
            imcol = i % 3
            pixrow = j // 112
            pixcol = j % 112
            t[imrow*114+pixrow][imcol*114+pixcol]=outa[i][j]
    newimage = Image.new('L', (340, 340))  # type, size
    # print(t)
    newimage.putdata(t.reshape((340*340,)))
    newimage.save("samples2.png")