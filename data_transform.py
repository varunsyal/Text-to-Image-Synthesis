
# coding: utf-8

# In[2]:


import numpy as np
from PIL import Image
import pickle
import torchvision.transforms as transforms
import torch


# In[3]:


image_transform_s1 = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

image_transform_s2 = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# In[4]:


def get_values(desc_fname, files_fname, text_dir, img_dir, stage=1):
    imgs = []
    emds = []
    text = []
    
    embed = np.load(desc_fname)
    embed_shape = embed.shape
    
    with open(files_fname,'rb') as file:
        dat = pickle.load(file)
    for i in range(len(dat)):
        for l in open(text_dir+dat[i]+'.txt','rb'):
            text.append(l.strip().decode('utf-8'))
        
        img = Image.open(img_dir+dat[i]+'.jpg').convert('RGB')
        if stage == 1:
            img_ = np.array(image_transform_s1(img))
        else:
            img_ = np.array(image_transform_s2(img))
        img_ = np.transpose(img_,(2,0,1))
        img_[0,:,:] = (img_[0,:,:]/255 - 0.5)/0.5
        img_[1,:,:] = (img_[1,:,:]/255 - 0.5)/0.5
        img_[2,:,:] = (img_[2,:,:]/255 - 0.5)/0.5
        imgs.append(img_)
        del img_
        
        for j in range(embed_shape[1]):
            emds.append(embed[i,j,:])
        
        if i%1000 == 0:
            print(i)
    
    return imgs, emds, text


# In[5]:


def read_input(dataset='CUB', stage=1):
    # Define File Destinations
    test_desc_fname = dataset+'/desc/test/char-CNN-RNN-embeddings.npy'
    train_desc_fname = dataset+'/desc/train/char-CNN-RNN-embeddings.npy'
    
    test_files_fname = dataset+'/desc/test/filenames.pickle'
    train_files_fname = dataset+'/desc/train/filenames.pickle'
    text_dir = dataset+'/desc/text_c10/'
    
    test_img_dir = dataset+'/images/'
    train_img_dir = dataset+'/images/'
    
    #Load Training Data
    print('Loading Training Data...')
    train_imgs, train_emds, train_text = get_values(train_desc_fname, train_files_fname, text_dir, train_img_dir, stage=stage)
    
    #Load Testing Data
    print('Loading Testing Data...')
    test_imgs, test_emds, test_text = get_values(test_desc_fname, test_files_fname, text_dir, test_img_dir, stage=stage)
        
    return train_imgs, train_emds, test_imgs, test_emds, train_text, test_text

