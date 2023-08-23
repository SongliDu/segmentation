import SimpleITK as sitk
import sys, os

# if len ( sys.argv ) < 3:
#     print( "Usage: DicomSeriesReader <input_directory> <output_file>" )
#     sys.exit ( 1 )

input_dir = "data/10000_1"
output_file = "data/10000_1.nii.gz"
print( "Reading Dicom directory:", input_dir )
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames( input_dir )
reader.SetFileNames(dicom_names)

image = reader.Execute()

size = image.GetSize()
print( "Image size:", size[0], size[1], size[2] )

print( "Writing image:", output_file )

sitk.WriteImage( image, output_file )

if ( not "SITK_NOSHOW" in os.environ ):
    sitk.Show( image, "Dicom Series" )