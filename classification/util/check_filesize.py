import os

def calculate_total_size(file_list):
    """
    Calculate the total size of all files in the given file list.

    Parameters:
        file_list (list): List of file paths.

    Returns:
        total_size (int): Total size of files in bytes.
    """
    total_size = 0
    for file_path in file_list:
        if os.path.isfile(file_path):  # Check if the path is a file
            total_size += os.path.getsize(file_path)
        else:
            print(f"File not found or not a valid file: {file_path}")
    return total_size

def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-separated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist

# Example usage:
file_list = read_file_list('/ailab/public/pjlab-smarthealth03/leiwenhui/Code/LesionAttribute_lwh/config/data/SynLesion/image_with_real_stomach_tumor.txt')

total_size_bytes = calculate_total_size(file_list)
total_size_gb = total_size_bytes / (1024 ** 3)  # Convert to gigabytes

print(f"Total size of files: {total_size_bytes} bytes ({total_size_gb:.2f} GB)")
