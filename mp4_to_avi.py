import os
import glob
import ntpath
import sys


def convert_mp4_to_avi(file_name, output_directory):
    input_name = file_name
    output_name = ntpath.basename(file_name)
    output = output_directory + output_name.replace('.mp4', '.avi', 1)
    cmd = 'ffmpeg -i "{input}" -c:v libx264 -c:a libmp3lame -b:a 384K "{output}"'.format(
        input=input_name,
        output=output)
    #    print(cmd)
    return os.popen(cmd)


if __name__ == '__main__':
    input_directory = "E://PycharmProjects/HealBone-GaitAnalysis/data/"
    output_directory = "E://PycharmProjects/HealBone-GaitAnalysis/data/"
    files = glob.glob(input_directory + '*.mp4')
    for file_name in files:
        try:
            print(file_name)
            convert_mp4_to_avi(file_name, output_directory)
        except:
            raise

# def main():
#     input_directory = sys.argv[1]
#     output_directory = sys.argv[2]
#     files = glob.glob(input_directory + '*.mp4')
#     for file_name in files:
#         try:
#             convert_mp4_to_avi(file_name, output_directory)
#         except:
#             raise
#
#
# if __name__ == "__main__":
#     main()
