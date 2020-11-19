# Pix2Pix-UI-tkinter
A GUI for pix2pix, easy train and test  
# How to use  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/UI-Design.png)
  First, all of your images have to pix2pix-image-format, like this:  
![image](https://github.com/KamiKazeFei/Pix2Pix-UI-tkinter/blob/main/format.png)  
you can use the combine.py the create pix2pix-image-format  
then you have to create a main folder, it have 5 subfolder:  
1.train : put the train images here  
2.test  : put the test  images here  
3.save  : it will save the model to here, complete model have include 3 file:  
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
5.
