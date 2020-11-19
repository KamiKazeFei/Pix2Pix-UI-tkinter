# Pix2Pix-UI-Tkinter
A GUI for pix2pix, easy train, and test  
# How to use  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/UI-Design.png)
  First, all of your images have to pix2pix-image-format, like this:  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/format.png)  
the right is input, the left is the target.  
you can use the combine.py the create pix2pix-image-format  
then you have to create the main folder, it has 5 subfolders:  
1.train: put the train images here  
2.test: put the test  images here  
3.save: it will save the model to here, the complete model includes 3 files:  
  ■ checkpoint  
  ■ *.index  
  ■ *.data-00000-of-00001  
  ■ * mean number, this number * 20 means the epochs that you trained.  
4.gt & gene: gt will save the target image, a gene will save the generated images.  
Before you start training, please be sure your test & train folder has an image and with the right format and .png  
  
then check your CUDA and cudnn can be used. Also, I set a GPU limit of 75% in main.py, if don't need it, just delete it.  
  
happy using.By kamikaze

# UI-introduce  
1.Show using CPU & GPU.  
  
2.Loading the main folder, the main folder has 5 subfolders:test, train, save, gt, gene.  
  
3.Show the path of the main folder.  
  
4.Confirm if "test" exists, if existed then show the number of images in the folder. else show "None".  
  
5.Check if "train" exists, if existed then show the number of images in the folder. else show "None".  
  
6.Check if "save" exists, if existed then show the newest model in the folder. if no model exists and s didn't found the folder, show "None". 
  
7.Check if "gt" & "gene" exists, if existed then show "OK". else show "None".  
  
8 & 9. Set epochs, type number in the textbox and click "Set epoch", then click "Start train" training will start.  
  
10 & 11.The set number that generatess, type number in the textbox and click "Set num", the click "Generate Image" generating will start.  
It will randomly take the image in the "test" folder, and will save the target image to "gt", generate an image to "gene" at the same time and the same name.  
  
12.QUIT UI  
  
13.Training preview.  
  
14.Record every epoch's time cost.  
    Show the process of generating.  
    Show the IQA number of gt & gene.  
  
15.Show the training process, use the star symbol.  
  
16.Show the UI-state.  
  
17.Cal the gt & gene IQA avg. like PSNR, SSIM, BRISQUE, NIQE.  
  



