import SimpleITK as sitk
import sys, os

input_dir = "/home/dusongli/project/CT_data/10001_1"
output_file = "/home/dusongli/project/segmentation/data/10001_1.nii.gz"
print( "Reading Dicom directory:", input_dir )
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames( input_dir )
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print( "Image size:", size[0], size[1], size[2] )

print( "Writing image:", output_file )

sitk.WriteImage( image, output_file )
