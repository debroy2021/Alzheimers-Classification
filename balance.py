import os
import albumentations as A
import cv2

#To cut the large folder down to 5000, we used os.remove()
def augmentImage(path):
  image = cv2.imread(path)
  transform = A.Compose([
    A.CLAHE(),
    A.Transpose(),
    A.HorizontalFlip(p=.5),
    A.Rotate(limit=30, p=.25),
    A.RandomBrightnessContrast(p=.5),
    A.RandomGamma(p=.5),
    ]) 
  augmented_image = transform(image=image)['image']
  return augmented_image

origPath = input("Enter the path for the dataset folder containing the 4 folders: ")

for path, dirs, files in os.walk(origPath + "/Moderate Dementia"):
  i=0
  for file in files:
    i+=1
    if i < 5000:
      filePath = os.path.join(path,file)
      for j in range(9):
        cv2.imwrite(os.path.join(origPath , 'aug_img'+ str(i) + " " + str(j) + ".jpg"), augmentImage(filePath))

for path, dirs, files in os.walk(origPath + "/Non Demented"):
  i=0
  for file in files:
    i+=1
    if i > 5000:
      filePath = os.path.join(path,file)
      os.remove(filePath)

for path, dirs, files in os.walk(origPath + "/Mild Dementia"):
  i=0
  for file in files:
    i+=1
    if i < 5000:
      filePath = os.path.join(path,file)
      os.remove(filePath)

for path, dirs, files in os.walk(origPath + "/Very mild Dementia"):
  i=0
  for file in files:
    i+=1
    if i < 5000:
      filePath = os.path.join(path,file)
      os.remove(filePath)