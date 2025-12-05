import os
import zipfile
import requests
import warnings

def read_zip(url, directory, filename = None):
    """
    Download a ZIP archive from a remote URL and unpack its files into a chosen directory.

    Parameters
    ----------
    url : str
        Web address pointing to the ZIP file that should be retrieved.
    directory : str
        Destination folder where all extracted files will be placed.
    filename : str, optional
        Name to assign to the downloaded ZIP file. If not provided, the url filename
        will be used.

    Returns
    -------
    None
        This function performs file download and extraction as side effects and
        does not return a value.
    """

    request = requests.get(url)
    if not filename:
        filename = os.path.basename(url)

    # check if URL exists, if not raise an error
    if request.status_code != 200:
        raise ValueError('The URL provided does not exist.')
    
    # check if the URL points to a zip file, if not raise an error  
    #if request.headers['content-type'] != 'application/zip':
    if filename[-4:] != '.zip':
        raise ValueError('The URL provided does not point to a zip file.')
    
    # check if the directory exists, if not raise an error
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')

    # write the zip file to the directory
    path_to_zip_file = os.path.join(directory, filename)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)

    # get list of files/directories in the directory
    original_files = os.listdir(directory)
    original_timestamps = []
    for filename in original_files:
        filename = os.path.join(directory, filename)
        original_timestamp = os.path.getmtime(filename)
        original_timestamps.append(original_timestamp)

    # extract the zip file to the directory
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory)

    # check if any files were extracted, if not raise an error
    # get list of files/directories in the directory
    current_files = os.listdir(directory)
    current_timestamps = []
    for filename in current_files:
        filename = os.path.join(directory, filename)
        current_timestamp = os.path.getmtime(filename)
        current_timestamps.append(current_timestamp)
    if set(current_files) == set(original_files):
        warnings.warn("The ZIP file is empty or nothing new was extracted. This could be due to a previous ZIP being downloaded again.", UserWarning)