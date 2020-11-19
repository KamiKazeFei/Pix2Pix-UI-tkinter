# Pix2Pix-UI-tkinter
A GUI for pix2pix, easy train and test  
# How to use  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/UI-Design.png)
  First, all of your images have to pix2pix-image-format, like this:  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/format.png)  
right is input, left is target.  
you can use the combine.py the create pix2pix-image-format  
then you have to create a main folder, it have 5 subfolder:  
1.train : put the train images here  
2.test  : put the test  images here  
3.save  : it will save the model to here, complete model include 3 file:  
  ■ checkpoint  
  ■ *.index  
  ■ *.data-00000-of-00001  
  ■ * mean number, this number * 20 means the epochs that you trained.  
4.gt & gene : gt will save the target image, gene will save the generate images.  

# UI-introduce  
1.Show using CPU & GPU.  
2.Loading main folder, main folder have 5 subfolder:test, train, save, gt, gene.  
3.Show the path of main folder.  
4.Confirm if "test" exists, if existed then show the number of images in folder. else show "None".  
5.Check if "train" exists, if existed then show the number of images in folder. else show "None".  
6.Check if "save" exists, if existed then show the newest model in folder. if no model exist and s didn't foundthe folder, show "None".  
7.Check if "gt" & "gene" exists, if existed then show "OK". else show "None".  
8 & 9.Set epochs, type number in textbox and click "Set epoch", the click "Start train" training will start.  
10 & 11.Set number that generate, type number in textbox and click "Set num", the click "Generate Image" generating will start.  
It will random take the image in "test" folder, and will save target image to "gt", generate image to "gene" at same time and same name.  
12.QUIT UI  
13.Training preview.  
14.Record every epochs time cost.  
   Show the process of generating.  
   Show the IQA number of gt & gene.  
15.Show the training process, use star symbol.  
16.Show the UI-state.  
17.Cal the gt & gene IQA avg. like PSNR, SSIM, BRISQUE, NIQE.  



