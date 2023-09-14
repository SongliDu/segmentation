import SimpleITK as sitk
import sys, os

# input_dir = "/home/dusongli/project/3Dircadb1 /3Dircadb1.1/MASKS_DICOM/rightlung"
# output_file = "/home/dusongli/project/segmentation/data/3Dircadb1/1_right_lung.nii.gz"
# print( "Reading Dicom directory:", input_dir )
# reader = sitk.ImageSeriesReader()

# dicom_names = reader.GetGDCMSeriesFileNames( input_dir )
# reader.SetFileNames(dicom_names)

# image = reader.Execute()

# size = image.GetSize()
# print( "Image size:", size[0], size[1], size[2] )

# print( "Writing image:", output_file )

# sitk.WriteImage( image, output_file )


reader = sitk.ImageSeriesReader()

root = "/home/dusongli/project/3Dircadb2"
output_path = "/home/dusongli/project/segmentation/data/3Dircadb/patient"
os.listdir(root)
for path, dirnames, filenames in os.walk(root):
    if (dirnames.__contains__('PATIENT_DICOM')):
        dicom_names = reader.GetGDCMSeriesFileNames( os.path.join(path, 'PATIENT_DICOM') )
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        output_file = os.path.join(output_path, path.split('/')[-1] + '.nii.gz')
        print( "Writing image:", output_file )

        sitk.WriteImage( image, output_file )

reader = sitk.ImageSeriesReader()



# root = "/home/dusongli/project/3Dircadb2"
# output_path_left = "/home/dusongli/project/segmentation/data/3Dircadb1/leftlung"
# output_path_right = "/home/dusongli/project/segmentation/data/3Dircadb1/rightlung"

# for path, dirnames, filenames in os.walk(root):
#     if (dirnames.__contains__('MASKS_DICOM')):
#         if (os.listdir(os.path.join(path, 'MASKS_DICOM')).__contains__('leftlung')):
#             dicom_names = reader.GetGDCMSeriesFileNames( os.path.join(path, 'MASKS_DICOM/leftlung') )
#             reader.SetFileNames(dicom_names)
#             image = reader.Execute()
#             output_file = os.path.join(output_path_left, path.split('/')[-1] + '.nii.gz')
#             print( "Writing image:", output_file )

#             sitk.WriteImage( image, output_file )

#             dicom_names = reader.GetGDCMSeriesFileNames( os.path.join(path, 'MASKS_DICOM/rightlung') )
#             reader.SetFileNames(dicom_names)
#             image = reader.Execute()
#             output_file = os.path.join(output_path_right, path.split('/')[-1] + '.nii.gz')

#             print( "Writing image:", output_file )

#             sitk.WriteImage( image, output_file )