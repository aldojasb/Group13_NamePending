# author: Son Chau
# date: 2021-11-19

"""This script prints out docopt args.
Usage: download_data.py --url=<arg1> --path=<arg2>

Options:
--url=<arg1>             URL from where to download the data (must be zip file)
--path=<arg2>            Path (not including filename) of where to locally write the file

Example:
python download_data.py --url=http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --path=./data
"""

from docopt import docopt
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

# define main function
def main():

    # parse arguments
    args = docopt(__doc__)

    # assign args to variables
    url = args['--url']
    path = args['--path']

    # download data
    with urlopen(url) as response:
        zf = ZipFile(BytesIO(response.read()))
        zf.extractall(path)

    print('Data downloaded to {}'.format(path))
 

# call main function
if __name__ == "__main__":
    main()
